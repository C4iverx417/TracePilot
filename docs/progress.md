# 开发与学习进度

状态流转：未开始 → 当前开发 → 代码已验证/等待用户研究 → 用户已复述/待巩固 → 用户明确授权下一步

| 单位 | 状态 | 备注 |
|---|---|---|
| U01 | 代码已验证 / 等待用户研究 | 生成模型暂用 mock（用户决定 API key 之后再定）；OpenAI 兼容客户端已用本地假服务器验证 |
| U02–U15 | 未开始 | |

---

## U01 学习交接卡

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
