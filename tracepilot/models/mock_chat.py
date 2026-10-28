"""An offline stand-in for a real chat model, speaking LangChain's BaseChatModel interface.

It is *extractive*: it reads the <evidence> blocks in the prompt, keeps those
that share words with the <question>, and answers with their first sentences,
citing the block ids. That is enough to exercise retrieval -> prompt -> parse ->
citation checks end to end without an API key. It is not a model of quality.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_EVIDENCE = re.compile(r'<evidence id="(?P<id>[^"]+)"(?:\s+\w+="[^"]*")*>\n(?P<body>.*?)\n</evidence>', re.S)
_QUESTION = re.compile(r"<question>\n?(?P<q>.*?)\n?</question>", re.S)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]{2,}")
_STOP = {"the", "and", "how", "what", "can", "does", "with", "for", "from", "that", "this", "are", "you", "use",
         "pydantic", "model", "models", "when", "why", "into", "not", "about", "which"}


def _terms(text: str) -> set:
    """Lower-cased words minus stop words, crudely stemmed to 5 letters ("immutable" ~ "immutability")."""
    return {w.lower()[:5] for w in _WORD.findall(text) if w.lower() not in _STOP}


def _coverage(question_terms: set, body: str) -> float:
    return len(question_terms & _terms(body)) / len(question_terms) if question_terms else 0.0


def _first_sentence(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "```", "!!!", "???", "|", "<", "[curated note", "- ")):
            sentence = re.split(r"(?<=[.!?])\s", line)[0]
            return sentence[:240]
    return body.strip().splitlines()[0][:240] if body.strip() else ""


class ExtractiveMockChatModel(BaseChatModel):
    max_findings: int = 2
    min_coverage: float = 0.5  # share of question terms that must appear in a block to "support" it

    @property
    def _llm_type(self) -> str:
        return "extractive-mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        match = _QUESTION.search(prompt)
        question = match.group("q") if match else ""
        q_terms = _terms(question)
        blocks = [(m.group("id"), m.group("body")) for m in _EVIDENCE.finditer(prompt)]
        overlapping = [(cid, body) for cid, body in blocks if _coverage(q_terms, body) >= self.min_coverage]
        overlapping.sort(key=lambda block: -_coverage(q_terms, block[1]))

        if not overlapping:
            payload = {
                "outcome": "insufficient_evidence",
                "summary": "[mock] None of the provided evidence mentions the terms in the question.",
                "findings": [],
                "hypotheses": [],
                "steps": [],
                "missing_information": ["Documentation that covers this question"],
            }
        else:
            chosen = overlapping[: self.max_findings]
            payload = {
                "outcome": "answered",
                "summary": "[mock] Extracted from the most relevant evidence; not a real model answer.",
                "findings": [{"text": _first_sentence(body), "citation_ids": [cid]} for cid, body in chosen],
                "hypotheses": [],
                "steps": [],
                "missing_information": [],
            }
        text = json.dumps(payload, ensure_ascii=False)
        usage = {"input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4,
                 "total_tokens": len(prompt) // 4 + len(text) // 4}
        message = AIMessage(content=text, usage_metadata=usage, response_metadata={"usage_source": "estimated"})
        return ChatResult(generations=[ChatGeneration(message=message)])
