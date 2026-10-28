# 开发与学习进度

状态流转（PRD v1.3 起）：未开始 → 开发中 → 已验证；U15 后交付 `docs/learning-guide.md`，用户统一研究。
U01、U02 按旧流程交付了学习交接卡（见下方），予以保留。

| 单位 | 状态 | 备注 |
|---|---|---|
| U01 | 已验证 | 生成模型暂用 mock；U02 中按 MVC 重组：`llm.py` → `models/llm.py`，`cli.py` 拆成 `controllers/cli.py` + `views/cli_view.py`，命令改为 `tracepilot ask` |
| U02 | 已验证 | 目标项目 Pydantic v2.14.1；本地 embedding（fastembed + multilingual MiniLM）；内存向量检索 |
| U03 | 已验证 | 基础 RAG：检索 → 提示词（证据块 + 规则）→ LangChain chat model → Pydantic schema 解析 → 引用校验；mock 为抽取式假模型 |
| U04 | 已验证 | 服务端确定性规则决定 outcome（answered / partially_answered / insufficient_evidence）；无检索结果不调用模型；可疑指令型资料标记；输出脱敏 |
| U05 | 已验证 | 两个版本 1.10.26 / 2.14.1（git tag 证据）；版本输入解析；先过滤后排序；unsupported_version / insufficient_context 不调用模型 |
| U06–U15 | 未开始 | |

---

## U01 学习交接卡

> U02 之后的运行方式：`uv run tracepilot ask "..."`；文件位置见上表备注。

**当前单位：** U01 —— 用户能在终端提问并获得模型回答

**实现范围：**
- 新增：`config.py`（读取环境变量/.env）、`llm.py`（mock 客户端 + OpenAI 兼容客户端，仅用标准库 `urllib`）、`cli.py`（命令行入口）、9 个单元测试。
- 已知限制：
  - 还没有真实模型调用过（没有 API key）。mock 的回答是占位文本，token 数是按"4 字符 ≈ 1 token"粗估的，标为 `(estimated)`。
  - 没有检索、没有多轮对话、没有重试（PRD 的退避重试留到 U08/U14）。
  - 中文的 token 粗估偏差很大，只是演示 usage 字段的位置。

**运行方式（均已在本机实际运行过）：**
```bash
python3 -m tracepilot "pip install 后 import 报错 ModuleNotFoundError 怎么办?"
python3 -m tracepilot --show-request "hello"          # 打印请求体
TRACEPILOT_PROVIDER=openai python3 -m tracepilot hi   # 缺 key → config error, exit 2
python3 -m tracepilot ""                              # 空问题 → INVALID_INPUT, exit 2
python3 -m unittest discover -s tests -v
```

**阅读路径：**
1. `tracepilot/cli.py` 的 `main()` —— 整条链路的起点。
2. `tracepilot/config.py` 的 `load_settings()` —— 配置从哪来、什么时候报错。
3. `tracepilot/llm.py` 的 `build_messages()` → `build_payload()` —— 消息 role 和请求体长什么样。
4. `tracepilot/llm.py` 的 `OpenAICompatibleClient.chat()` —— 真正的 HTTP 请求和错误分类。
5. `tracepilot/llm.py` 的 `parse_response()` —— 从返回 JSON 里取 `choices[0].message.content` 和 `usage`。
6. `tests/test_u01.py` 的 `FakeProvider` —— 没有 key 时怎么验证真实请求路径。

**数据流：**
```
命令行参数 → 输入校验(空/超长) → load_settings(env + .env)
 → build_messages: [system, user] → make_client(provider)
 → mock: 本地拼字符串 | openai: POST {base_url}/chat/completions (Authorization 头带 key)
 → parse_response: text / model / finish_reason / usage → render 打印
```

**验证结果：**
| 场景 | 结果 |
|---|---|
| mock 正常提问 | 返回占位回答 + 元信息，exit 0 |
| `--show-request` | 打印请求 JSON，不含 key |
| provider=openai 且无 key | `config error: ... OPENAI_API_KEY is not set`，exit 2 |
| 空问题 | `error [INVALID_INPUT]`，exit 2 |
| 假 key 打真实 OpenAI | `error [MODEL_AUTH_FAILED] ... (HTTP 401)`，exit 1，未打印 key |
| 假服务器：成功 / 401 / 超时 / 返回格式错 | 9 个单测全部通过 |
| **未验证** | 真实模型返回的回答质量、真实 usage、真实延迟 |

**研究问题（先自己想，再来跟我讲）：**
1. `system` 和 `user` 两条消息分别起什么作用？如果把 `SYSTEM_PROMPT` 删掉，请求体会变成什么样？
2. API key 出现在请求的哪个位置？代码里做了哪几处措施保证它不会被打印出来？
3. `ModelAuthError`、`ModelTimeoutError`、`InvalidModelOutputError` 分别在什么条件下抛出？为什么 401 要和 500 区分开？
4. `Usage.source` 为什么要区分 `provider` / `estimated` / `None`，而不是没拿到就记成 0？（提示：PRD FR-11）
5. 为什么"调用成功、exit 0"不代表答案是正确的？

**动手任务：** 把 `.env` 里的 `TRACEPILOT_TEMPERATURE` 或 `TRACEPILOT_MAX_TOKENS` 改一下，用 `--show-request` 观察请求体的变化；或者给 `render()` 多输出一个字段（比如 `base_url`），并解释为什么 mock 模式下回答内容不会随 temperature 变化。

**停止状态：** 代码验证完成，等待用户研究；**未开始 U02**。

---

## U02 学习交接卡

**当前单位：** U02 —— 用户能从少量资料中找出相关原文

**实现范围：**
- 资料：Pydantic **v2.14.1**（git tag，commit `b2212c7`）的 5 份文档：Models、Validators、Configuration、Error Handling、Migration Guide。原文快照存在 `data/sources/pydantic/snapshot/`，`manifest.json` 记录来源与版本证据，`fetch_log.json` 记录抓取时间与 SHA-256。
- 新增：快照加载与哈希校验、按标题分块（记录精确字符偏移）、本地 embedding、LangChain `InMemoryVectorStore` 检索、`search` / `devcheck` 命令、15 个新测试。
- 按 PRD 新增的第 10.3 节重组为 MVC：`models/`、`views/`、`controllers/`，对象组装集中在 `app.py`。
- 环境：改用 uv + Python 3.12，依赖锁定在 `uv.lock`。
- 已知限制：
  - 检索结果**总会返回 top-k**，哪怕问题与资料完全无关（u02-05）。U02 不能拒答，拒答在 U04。
  - 只有一个版本；版本过滤在 U05。没有关键词检索（U06）。
  - 索引只在内存中，每次运行重新计算 embedding（约 5–8 秒）。
  - embedding 模型只读每个分块的**前 128 个 token**（见下方验证结果）。

**运行方式（均已在本机实际运行过）：**
```bash
uv sync
uv run tracepilot search "How can I make a Pydantic model immutable?" -k 3
uv run tracepilot search -k 8 --full "如何在字段校验之前先对原始输入做转换？"
uv run tracepilot search                     # 交互模式，索引只建一次
uv run tracepilot devcheck                   # 5 道开发问题
TRACEPILOT_MANIFEST=nope.json uv run tracepilot search x   # 失败：config error, exit 2
uv run python -m unittest discover -s tests -v
```

**阅读路径：**
1. `tracepilot/controllers/cli.py` 的 `cmd_search()` → `_build_index()` —— 入口。
2. `tracepilot/app.py` 的 `build_retriever()` —— 一条离线链路被组装起来的地方，先读它能看到全貌。
3. `tracepilot/models/sources.py` 的 `load_snapshot()` —— 资料怎么变成 `SourceDocument`，版本证据从哪来。
4. `tracepilot/models/chunking.py` 的 `_sections()` 与 `_pack()` —— 分块规则（重点看代码块里的 `#` 为什么不能当标题）。
5. `tracepilot/models/retrieval.py` 的 `embedding_text()`、`index()`、`search()` —— 被 embedding 的到底是什么文本、分数怎么来。
6. `tracepilot/models/embeddings.py` —— 一个适配器如何让 fastembed 符合 LangChain 的 `Embeddings` 接口。
7. `tracepilot/views/cli_view.py` 的 `render_hits()` —— 输出里每个字段的含义。

**数据流：**
```
离线（每次启动时）:
manifest.json + snapshot/*.md → load_snapshot (校验 SHA-256) → SourceDocument ×5
 → MarkdownChunker.split (按标题切节 → 超长节按段落打包，≤400 字符) → Chunk ×533
 → embedding_text = "文档标题 > 标题路径" + 原文 → fastembed → 384 维向量 → InMemoryVectorStore

在线（每个问题）:
问题 → 同一模型 embed_query → 与 533 个向量算余弦相似度 → 取 top-k
 → SearchHit(rank, score, chunk, document) → render_hits → 终端
```

**MVC / SOLID 在本单位的体现（可用于面试讲解）：**
- **M** `models/*` 只做数据与逻辑，不 print；**V** `views/cli_view.py` 只把结果变成字符串；**C** `controllers/cli.py` 只做参数解析、校验、调用、异常到退出码的映射。
- **依赖倒置**：`VectorRetriever` 依赖 LangChain 的 `Embeddings` 抽象，不认识 fastembed；具体用哪个由 `app.py` 决定。
- **里氏替换**：测试里把 `FastEmbedEmbeddings` 换成 `DeterministicFakeEmbedding`，检索器代码不变。
- **开闭**：以后换 OpenAI embedding 只需新增一个 `Embeddings` 实现并在 `app.py` 加一个分支。
- **克制**：没有为"检索器"建抽象接口——目前只有一个实现，按 PRD 10.3 不预建；U06 加关键词检索时再提取。

**验证结果：**

`devcheck`（top-5，chunk size 400）：

| 题目 | 类型 | 结果 | 说明 |
|---|---|---|---|
| u02-01 如何让模型不可变 | direct | **MISS** | 正确章节 "Faux immutability" 排第 7；top-1 是 Models 开头的泛泛介绍（0.725） |
| u02-02 `BaseSettings` 报错原文 | exact_error | PASS（第 1） | |
| u02-03 V1→V2 validator 变化 | multi_chunk | PASS（5/5 都来自 "Changes to validators"） | 800 字符时只有 2/5 |
| u02-04 中文问"校验前转换输入" | cross_lingual | PASS（第 1、3） | 中文检索英文资料可行，但分数整体偏低（0.47） |
| u02-05 Django CORS | no_answer | N/A | 依旧返回 5 个结果，top-1 = 0.343 |

可回答题 3/4 命中——只是 5 道题的冒烟检查，**不是准确率**。

分块大小的实测（同一 tokenizer 统计）：

| chunk size | 分块数 | token 中位数 | 超过 128 token 被截断 |
|---|---|---|---|
| 800 | 288 | 182 | 202 / 288（70%） |
| 400（当前默认） | 533 | 98 | 112 / 533（21%） |

默认值从 800 改为 400 的依据是模型的 128 token 输入上限，而不是为了让考题通过（两种设置下都是 3/4）。

**未验证：** 更大规模问题集上的检索质量；其他 embedding 模型的效果；U02 不涉及生成，所以没有回答质量可言。

**研究问题（先自己想，再来跟我讲）：**
1. 为什么分块器要跟踪"是否在 ``` 代码块里"？如果不跟踪，Pydantic 文档里的 `#> output` 会造成什么后果？
2. `embedding_text()` 把"文档标题 > 标题路径"拼到原文前面。这对 u02-03 这种问题有什么帮助？会不会有副作用？
3. u02-05（Django CORS）top-1 只有 0.343，u02-04（中文问题）top-1 也只有 0.470。能不能简单设一个"低于 0.4 就算没答案"的阈值？为什么？
4. u02-01 为什么没命中？"immutable" 和文档里的 "immutability"、"frozen" 是什么关系？你觉得 U06 的关键词检索能不能解决它？
5. 为什么 `Chunk` 要存 `char_start/char_end` 而不只是存 `content`？这对 PRD 的"引用可核对"有什么意义？

**动手任务：** 用 `TRACEPILOT_CHUNK_SIZE=800` 和 `=200` 各跑一次 `uv run tracepilot devcheck`，再用 `-k 10` 跑一次，对比 u02-01 和 u02-03 的排名变化，并解释原因（变化可能不大，不要强行归因）。

**停止状态：** 代码验证完成，等待用户研究；**未开始 U03**。

---

## 单位记录（PRD v1.3 起，不再附研究问题）

### U03 —— 用户能根据资料得到带来源的回答（已验证）

- **新增：** `models/answering.py`（AnswerDraft schema、提示词、引用校验）、`models/chat.py`（调用 LangChain chat model 并把 openai 异常映射成项目错误码）、`models/mock_chat.py`（抽取式 mock，继承 `BaseChatModel`）、`models/rag.py`（检索一次、生成一次）、`models/runlog.py`（每次运行写 `var/runs/<run_id>.json`）、`tracepilot answer` 命令。
- **关键设计：** 模型只返回引用 ID；标题、URL、版本、片段由服务端从证据补全；不在证据里的 ID 被删除并在输出中提示（AT-11）。证据块用 `<evidence>` 包裹并转义，防止资料内容伪造块边界。真实模型用 `ChatOpenAI`，`max_retries=0`，重试留给工作流（U08）。
- **验证：** 34 个测试通过。mock 下：“immutable” 问题（k=8）引用 Models > Faux immutability；V1→V2 validator 问题引用两段 Changes to validators；Django CORS 返回 insufficient_evidence。
- **调试记录：** 证据块属性 `source="Models > ..."` 中的 `>` 曾让 mock 的正则提前截断，导致全部误判为证据不足；已修正正则。
- **限制：** mock 只按词重叠抽取首句，回答质量无意义；真实模型未验证。成本记录为 null（未知价格不计 0）。

### U04 —— 用户能知道资料不足以回答（已验证）

- **新增：** `models/guards.py`：`enforce_outcome`（无有效引用的结论移出 findings；answered 中有无支持内容 → partially_answered；一条有支持的都没有 → insufficient_evidence）、`suspicious_evidence`（正则识别“忽略之前指令 / 泄露密钥 / 执行命令”等文本）、`redact_secrets`（`sk-…`、`ghp_…`、`AKIA…`）。`RagPipeline.generate` 在检索为空时直接返回 insufficient_evidence，不调用模型。
- **关键设计：** 模型自报的 outcome 只能被服务端**降级**，不能升级；“引用存在”由代码校验，“引用是否真的支持结论”仍需人工或评估（U15）。提示词规则 3 要求把资料中的指令当作内容描述。
- **验证：** 42 个测试通过，固定样例：可回答 → answered 且有引用；知识库外 → insufficient_evidence 且无引用；恶意资料 → 被标记、输出提示“treated as data”。另测：模型捏造 `c99` 的“answered”被降级为 insufficient_evidence。
- **限制：** 指令识别是关键词正则，只能覆盖常见写法；mock 仍可能把恶意句子当作“文档内容”引用（描述而非执行），真实模型的遵守情况未验证。仅一次检索，不补检索（U08）。

### U05 —— 用户能获得适用于自己版本的回答（已验证）

- **数据：** 新增 Pydantic **v1.10.26**（commit `d771df9`）4 份文档：Models、Validators、Model Config、Settings management。目录改为 `data/sources/pydantic/<version>/`，`project.json` 列出所有版本快照。v1 文档中的代码示例是构建期 include（`{!.tmp_examples/...!}`），未进入快照。
- **新增：** `models/versions.py`（版本输入规则）、`sources.load_project`、`VectorRetriever.search(..., version=)`、`RagPipeline.version_gate`、`answer --version`、`search --version`、`devcheck --set u05`。
- **版本规则：** `2.14.1` / `2.14` / `v1.10` → 精确匹配；`unknown`、空、`latest`、`最新版`、只有主版本号 `2` → insufficient_context（不默认为最新版）；格式正确但未覆盖（`2.5`）→ unsupported_version；格式错误（`abc`）→ INVALID_INPUT。后两类以及 unknown 都**不检索、不调用模型**。
- **关键设计：** 版本过滤通过 `InMemoryVectorStore` 的 `filter` 在计算相似度**之前**剔除不匹配的块（已读源码确认），而不是靠提示词让模型“注意版本”。每个块在向量库的 metadata 里带 `version_scope`。
- **验证：** 52 个测试通过。`devcheck --set u05`：3 组同题不同版本共 6 题，6/6 命中对应版本的章节，**错误版本块 0 个**；`devcheck`（u02 题集，现在固定 2.14.1）仍为 3/4。对照：不加过滤时 “BaseSettings 读取环境变量” 的 top-5 **全部来自 v1**，v2 用户会被误导。同一问题 `--version 1.10` 引用 `allow_mutation = False`，`--version 2.14.1` 引用 v2 的 Faux immutability。
- **限制：** 没有“通用内容”（`general`）标注；跨版本对比（升级问题）留到 U07；unknown 时一律追问，尚未判断问题是否真的依赖版本（U09）。
