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
| U06 | 已验证 | BM25（自实现）+ 向量，RRF 融合；`Retriever` 接口；`--retrieval` 可切换基线；默认 top_k=8 |
| U07 | 已验证 | `upgrade --from --to`：两版本分别检索、分组呈现；5 个人工筛选的真实 Issue（GitHub 角色 + 解决证据）；文档事实 / 历史案例分离；冲突机制；每文档最多 2 块 |
| U08 | 已验证 | LangGraph 状态图：check_input → retrieve → assess →（补检索 / generate）→ finish；预算：3 轮检索、8 次调用、1 次格式修复、2 次瞬时重试、60 秒；`--trace` |
| U09 | 已验证 | `tracepilot chat`：内存会话；Session / Episode / Run；最多 2 轮澄清、每次 ≤3 个字段；显式版本优先；改版本即按新版本重新检索；会话隔离 |
| U10 | 已验证 | FastAPI：202 + 轮询；后台线程池（最多 3 并发）；每会话 1 个活动任务（409）；episode 校验；引用详情；统一错误体 |
| U11–U15 | 未开始 | |

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

### U06 —— 用户能用错误原文找到排查依据（已验证）

- **新增：** `models/keyword.py`（BM25，公式写在文件头；标识符不拆分，`allow_mutation` 是一个词）、`HybridRetriever`（RRF：`1/(60+rank)` 求和，只用排名，不需要两种分数同量纲）、`Retriever` Protocol（有了第二个实现才抽象）、`SearchHit.channels`（显示每块来自 `vector#3`、`bm25#1` 等）、`--retrieval hybrid|vector|keyword`、devcheck 输出 MRR。
- **参数：** 每路候选 20，融合去重后取 top_k=8 进上下文（PRD FR-04，默认值由 5 改为 8）。BM25 的 IDF 在**版本过滤后的集合**上计算。
- **验证（top-8，60 个测试通过）：**

| 题集 | vector | keyword (BM25) | hybrid |
|---|---|---|---|
| u06（5 精确词 + 5 语义改写） | 10/10，MRR 0.702 | 9/10，MRR 0.667 | 9/10，MRR **0.833** |
| u05（版本对） | 6/6，MRR 0.722 | 6/6，MRR 0.711 | 6/6，MRR **0.833** |
| u02 | 4/4，MRR 0.786 | 2/4，MRR 0.542 | 4/4，MRR **0.833** |

  所有配置错误版本块均为 0。
- **解读：** 混合检索把正确块**排得更靠前**（如 u06-s1 不可变问题：向量第 7 → 混合第 3；u06-s5：第 8 → 第 1），但 u06-s4（“validator 抛 type error 还算不算 validation error”）向量排第 4，BM25 前 20 没有它，融合后被两路都命中的块挤出 top-8——RRF 的已知弱点。BM25 对中文问题完全无效（无共同词），全靠向量通道。题量 20 道，**不能据此宣称混合检索更好**（PRD 13.2）。
- **限制：** 无重排器（P1）；没有调 RRF 常数和候选量。

### U07 —— 用户能了解一次版本升级的变化与排查步骤（已验证）

- **数据：** `data/sources/pydantic/issues/`：5 个已关闭 Issue（#6777、#7472、#6521、#6181、#11538），`scripts/fetch_issues.py` 抓取正文与全部评论，每条评论标注 GitHub `author_association`（MEMBER / CONTRIBUTOR / NONE），**不凭语气判断维护者**。`manifest.json` 中人工标注：版本范围（如 `range:>=2.0.0,<3.0.0`；#6521 只验证到 2.0.2，故不会作为 2.14.1 的证据）、解决状态（answered / fixed / not_planned——#11538 关闭但“不计划修复”，**closed ≠ solved**）、解决证据评论链接、根因备注。
- **新增：** `IssueMeta`、`load_issue_snapshot`、版本范围 `in_range`、`RagPipeline.compare / prepare_compare`、`guards.split_historical`（只被 Issue 支持的结论 → “相似历史案例”，outcome 最多 partially_answered）、`guards.find_conflicts`（人工标注 `conflicts_with` 的 Issue 与文档同时被引用 → conflicting_evidence）、`retrieval.diversify`（每文档最多 2 块）、`tracepilot upgrade`。证据块新增 `type` 和 `group` 属性，Issue 证据带人工标注的 `[curated note]`。
- **调试记录：**
  1. GitHub 正文含 `\r\n`，`read_text` 自动转换换行导致哈希不符；改为按字节读取 + 抓取时统一换行。
  2. 升级问题首次运行时，#7472 的标题与报错原文**完全相同**，加上标题在嵌入文本里重复两次，导致 v2 组 4 个名额全被这一个 Issue 占据——而它恰好是“根因不同”的案例，迁移指南被挤出。修正：嵌入文本去掉重复标题；组装上下文时每文档最多 2 块（只影响上下文选择，不改变检索排名，devcheck 结果不变）。修正后 v2 组为：#7472 ×2、迁移指南、#6777。
- **验证：** 67 个测试通过（含 AT-07 分组、AT-10 仅相似 Issue 不断言根因、冲突、范围版本）。devcheck 三个题集结果不变（u02 4/4、u05 6/6、u06 9/10，错误版本块 0）。
- **限制：** 5 个 Issue 远少于 PRD 规划的 30–50 个；真实数据中**没有**已标注的文档冲突，冲突机制只由测试夹具验证；mock 只抽取首句，无法真正“比较”两组证据。

### U08 —— 首轮检索不足时得到有限补检索后的结果（已验证）

- **新增：** `models/workflow.py`（`SupportWorkflow`、`WorkflowState`、`Budget`、`EvidenceAssessment` schema）；`rag.finish_answer`（把调用后的确定性检查从 `RagPipeline.generate` 中拆出，工作流和基线共用）；`ModelError.retryable`（超时、429、5xx、连接错误可重试；鉴权、格式错误不重试）；`answer --engine graph|rag`、`--max-rounds`、`--trace`；配置 `TRACEPILOT_ENGINE / MAX_ROUNDS / MAX_MODEL_CALLS / DEADLINE_S`。图结构导出在 `docs/workflow.mmd`。
- **图：** `check_input`（版本门）→ `retrieve`（合并进候选池）→ `assess`（模型按 schema 判断是否充分、哪些相关、给出改写查询）→ 条件路由：充分 → `generate`；不充分且有新查询且未到轮数上限 → 回到 `retrieve`；否则 → `generate`（用现有证据回答或说明缺口）→ `finish`。`trace` 字段用 `operator.add` reducer，每个节点追加一条事件。
- **硬约束在代码里：** 每次真正发出的请求（含重试、修复）都计入调用数；超出调用预算 → `BUDGET_EXCEEDED`，超时 → `DEADLINE_EXCEEDED`；格式错误只修复一次，再错 → `INVALID_MODEL_OUTPUT`；系统故障通过 `error_code` 表达，**不伪装成“证据不足”**（FR-08）。未经评估确认充分的 “answered” 会降级为 partially_answered。
- **调试记录：** 候选池最初按分数合并，但 BM25 分数**跨查询不可比**，第一轮噪声的高分压过第二轮的正确结果；改为按“最佳排名，同排名取较新一轮”合并。
- **验证：** 78 个测试通过：AT-08（首轮无关、改写后命中、3 次调用）、AT-09（3 轮上限、绝无第 4 轮）、改写与原查询相同时提前停止、一次格式修复成功 / 两次失败、429 退避重试（1s、2s）、重试上限、鉴权错误不重试、调用预算、截止时间（假时钟）。CLI 上 Django 问题：2 轮检索、3 次调用后结束，end_reason=retrieval_budget_exhausted。
- **限制：** mock 的评估只看词覆盖率，真实模型的判断质量未验证；`upgrade` 仍走单次 RAG（未迁入图）；`--show-context` 使用单次 RAG 引擎。

### U09 —— 用户能补充缺失信息后继续同一个问题（已验证）

- **新增：** `models/sessions.py`（`Session`、`Episode`、`ConfirmedContext`、`Message`、`SessionStore` 接口 + `InMemorySessionStore`）；`models/conversation.py`（`ConversationService.submit / new_session / new_episode`、`extract_conditions`、`is_vague`、`compose_query`）；`tracepilot chat`（`/new`、`/version`、`/session`、`/newsession`、`/switch`）。
- **规则：** 缺版本 → 问版本；问题含糊（短、含“失败 / 报错 / doesn't work”等）且没有错误原文 → 问错误原文；每次最多问 3 个字段，已确认的不重复问。澄清计数属于 **episode**，跨 run 累加；满 2 轮后不再追问：仍缺版本 → `insufficient_context` 并列出缺口；只缺错误原文 → 带提示尝试回答。只有 `/new`（新问题）或新会话会重置计数。优先级：显式选择（`--version` / `/version`）> 用户文字陈述 > 已有值；只有“pydantic 2.14 / 我用的是 1.10 / version 2.0”这类明确陈述算数，URL 或 traceback 里的版本号不算。
- **上下文预算：** 工作流只收到 `compose_query` 生成的紧凑查询（原问题 + 错误原文 + 最近 2 条补充，去掉版本陈述，≤2000 字符），从不拼接完整对话历史。
- **验证：** 88 个测试通过（AT-04 不猜版本、AT-05 两轮后停止、AT-12 改版本后证据全部来自新版本、AT-13 两个会话互不影响、显式选择优先、新问题重置计数但保留所选版本）。CLI 演示：“程序启动失败” → 问版本 + 错误原文 → “我用的是 2.14.1” → 只再问错误原文 → 贴错误 → 回答引用迁移指南并把 #6777 列为历史案例。
- **限制：** 会话只在内存中，**退出进程即丢失**（U13 持久化）；条件提取与“含糊”判断是确定性规则，不用模型，覆盖面有限；环境字段已建模但尚未提取。

### U10 —— 用户能通过 API 完成一次排查（已验证）

- **新增：** `controllers/api.py`（FastAPI 路由与错误映射）、`views/api_schemas.py`（请求/响应 schema 与映射函数，HTTP 契约集中在一处）、`models/runs.py`（`RunRegistry` 加锁的运行登记、`RunExecutor` 线程池执行、`STAGES` 节点 → 用户可见阶段）、`SupportWorkflow.run(on_progress=)`（用 LangGraph `stream(stream_mode="values")` 在每个节点完成后上报真实进度）、`app.build_api_context`、`tracepilot serve`（默认只绑定 127.0.0.1）。
- **接口：** `GET /health`、`GET /api/v1/projects/current`、`POST /api/v1/sessions`、`GET /api/v1/sessions/{id}`、`POST /api/v1/sessions/{id}/episodes`、`POST /api/v1/sessions/{id}/runs`（202 + `status_url`）、`GET /api/v1/runs/{id}`、`GET /api/v1/runs/{id}/citations/{cid}`。交互式文档在 `/docs`。
- **错误：** 统一为 `{"error": {"code", "message", "run_id"}}`：422 `INVALID_INPUT`（空问题、超长、非法版本、日志 > 20,000 字符）、404 `NOT_FOUND`、409 `RUN_IN_PROGRESS`（附现有 run_id）、409 `EPISODE_MISMATCH`（只能针对当前问题提交）。模型故障 → run `status=failed` + 错误码，`outcome` 为空，不伪装成“没有资料”。引用详情只对**本次 run 已校验的引用**开放。
- **验证：** 95 个测试通过（用一个可控阻塞的假模型：模型阻塞时轮询 < 1 秒返回、同会话第二次提交 409 而其他会话不受影响、澄清走 HTTP、过期 episode、5 类非法输入、鉴权失败 → failed）。另用 curl 对真实服务（真实 embedding + mock 模型）走通：建会话 → 含糊问题 → awaiting_input → 补充版本与错误日志 → completed、引用详情、伪造引用 404。
- **限制：** 运行登记与会话都在内存中，**服务重启即丢失**（U13）；没有取消和幂等键（U14）；无身份认证，只适合本机。
