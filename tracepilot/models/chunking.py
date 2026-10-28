"""Heading-aware Markdown chunking.

Rules, in order:
1. Split the document at Markdown headings (ignoring '#' lines inside ``` code
   fences, e.g. Pydantic's '#> output' comments). Each section remembers its
   heading path, like ("Migration guide", "Changes to validators").
2. A section that fits in `chunk_size` characters becomes one chunk.
3. A longer section is packed paragraph by paragraph (blank-line separated,
   never splitting inside a code fence) into chunks of at most `chunk_size`.
   A single paragraph longer than that is cut by characters as a last resort.

Every chunk stores exact character offsets into the snapshot file, so its text
can always be traced back to the original source.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .documents import Chunk, SourceDocument, sha256

PARSER_VERSION = "md-heading-v1"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

Span = Tuple[int, int]  # [start, end) character offsets


class MarkdownChunker:
    def __init__(self, chunk_size: int = 400):
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        self.chunk_size = chunk_size

    def split(self, doc: SourceDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        for heading_path, (start, end) in self._sections(doc.text):
            for piece_start, piece_end in self._pack(doc.text, start, end):
                content = doc.text[piece_start:piece_end]
                if not content.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}#{len(chunks):03d}",
                        document_id=doc.document_id,
                        heading_path=heading_path,
                        content=content,
                        char_start=piece_start,
                        char_end=piece_end,
                        content_hash=sha256(content),
                        version_scope=doc.version_scope,
                        parser_version=PARSER_VERSION,
                    )
                )
        return chunks

    # ------------------------------------------------------------ internals

    @staticmethod
    def _lines(text: str):
        """Yield (line, start_offset, inside_code_fence) for each line."""
        offset, in_fence = 0, False
        for line in text.splitlines(keepends=True):
            is_fence = bool(_FENCE.match(line))
            yield line, offset, in_fence or is_fence
            if is_fence:
                in_fence = not in_fence
            offset += len(line)

    def _sections(self, text: str) -> List[Tuple[Tuple[str, ...], Span]]:
        sections = []
        stack: List[Tuple[int, str]] = []  # (level, title)
        current_path: Tuple[str, ...] = ()
        current_start = 0
        for line, offset, in_code in self._lines(text):
            match = None if in_code else _HEADING.match(line)
            if not match:
                continue
            if offset > current_start:
                sections.append((current_path, (current_start, offset)))
            level, title = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = tuple(t for _, t in stack)
            current_start = offset
        if current_start < len(text):
            sections.append((current_path, (current_start, len(text))))
        return sections

    def _pack(self, text: str, start: int, end: int) -> List[Span]:
        if end - start <= self.chunk_size:
            return [(start, end)]

        # Paragraph boundaries: blank lines outside code fences.
        blocks: List[Span] = []
        block_start = start
        for line, offset, in_code in self._lines(text[start:end]):
            if not in_code and not line.strip():
                cut = start + offset + len(line)
                blocks.append((block_start, cut))
                block_start = cut
        if block_start < end:
            blocks.append((block_start, end))

        pieces: List[Span] = []
        piece_start, piece_end = blocks[0][0], blocks[0][0]
        for b_start, b_end in blocks:
            if b_end - piece_start <= self.chunk_size:
                piece_end = b_end
                continue
            if piece_end > piece_start:
                pieces.append((piece_start, piece_end))
            piece_start = b_start
            # An oversized single block: hard-cut it.
            while b_end - piece_start > self.chunk_size:
                pieces.append((piece_start, piece_start + self.chunk_size))
                piece_start += self.chunk_size
            piece_end = b_end
        if piece_end > piece_start:
            pieces.append((piece_start, piece_end))
        return pieces
