# RAG-Lite 沙箱验证平台 PRD

## 1. 产品定位

RAG-Lite 是一个本地轻量化 RAG 验证平台，用于在业务私有文档上验证检索、切片、重排、生成和评测口径的质量。它服务于业务专家、产品经理和算法/工程同学，强调可解释的实验闭环，而不是单纯的聊天体验。

核心目标：

- 让业务文档在本地完成解析、切片、索引和检索验证。
- 让用户能比较不同检索参数、切片参数和模型组合的效果。
- 通过评测集、RUN、baseline、错误归因和诊断面板，定位 RAG 失败原因。
- 在本地资源有限、Ollama 不稳定时，仍能用 retrieval-only 模式完成召回评估。

非目标：

- 不做生产级权限系统。
- 不做审计日志、脱敏导出、模型版本强绑定等部署治理能力。
- 不替代企业级 RAG 服务端或知识库平台。

## 2. 用户流程

### 2.1 知识库构建

1. 用户上传 PDF / TXT / Markdown / DOCX。
2. 用户选择嵌入模型、切分策略、chunk size、overlap。
3. 系统解析文档，必要时执行 OCR / 视觉增强。
4. 系统构建临时 Chroma 索引。
5. 自检通过后激活为版本化索引。
6. UI 展示文件状态、切片预览、切片统计和索引运维信息。

### 2.2 对话验证

1. 用户选择 LLM、Top-N、Top-K、rerank、num_ctx、系统提示词。
2. 系统执行统一检索管线。
3. 有上下文时调用 LLM 生成答案。
4. 无上下文时直接拒答，不调用 LLM。
5. UI 展示参考来源和检索诊断。
6. 用户可打分、备注、导出 QA 日志。

### 2.3 批量回放

1. 用户输入多行问题。
2. 系统逐条执行检索和生成。
3. 结果写入新会话。
4. 实验对比面板按参数聚合已有 QA 记录。

### 2.4 评测运行

1. 用户导入评测集；同名导入生成新的版本化数据集，不覆盖历史 RUN。
2. 系统运行 retrieval-only 或完整 LLM 评测。
3. 每个 case 保存检索诊断、命中判断、错误类型和归因。
4. RUN 保存参数快照、run fingerprint 和 summary。
5. UI 展示总览、结果明细、诊断分析、治理项和 RUN 对比。

### 2.5 评测治理

1. 系统扫描评测集质量问题。
2. 系统列出 unresolved expected file。
3. 用户在 alias 面板中确认 expected raw 到真实文件名的映射。
4. 后续评测自动使用 alias。
5. 用户可设置 baseline，并将后续 RUN 与 baseline 对比。

## 3. 功能需求

### FR-1 知识库构建与索引运维

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-1.1 | 支持多格式文档导入 | 已实现。支持 PDF / TXT / Markdown / DOCX。 |
| FR-1.2 | 支持切分参数配置 | 已实现。支持 sentence / token / paragraph，支持 chunk size 和 overlap。 |
| FR-1.3 | 支持本地向量索引 | 已实现。使用 Chroma PersistentClient，本地持久化。 |
| FR-1.4 | 索引构建具备失败保护 | 已实现。临时目录构建，自检通过后版本化激活；失败保留上一可用版本。 |
| FR-1.5 | 上传文件状态可见 | 已实现。展示已入库、新增、已变更、嵌入模型不一致等状态。 |
| FR-1.6 | 支持切片预览 | 已实现。按文件拉取 Chroma chunk、metadata 和正文片段。 |
| FR-1.7 | 支持切片统计 | 已实现。展示文件数、chunk 数、平均字符数、空块、OCR/视觉提示块等。 |
| FR-1.8 | 支持索引运维诊断 | 已实现。展示 active Chroma 目录、manifest 指向、版本数量、building 残留、磁盘占用。 |
| FR-1.9 | 支持旧索引清理预览 | 已实现。支持 dry-run 清理旧版本和 residual building，默认保护 active 和最近历史版本。 |
| FR-1.10 | 支持 OCR/视觉增强 | 已实现。PDF/DOCX 可选 OCR 和 Ollama 视觉补充，默认关闭以保证主流程稳定。 |

### FR-2 检索与重排

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-2.1 | 向量召回 | 已实现。使用 LlamaIndex + Chroma。 |
| FR-2.2 | 关键词/BM25 召回 | 已实现。纯 Python 本地实现，支持 CJK 字符、bigram、trigram tokenization。 |
| FR-2.3 | 混合召回 | 已实现。vector + keyword 合并去重，记录来源和分数。 |
| FR-2.4 | 可选 rerank | 已实现。支持 CrossEncoder rerank。 |
| FR-2.5 | 统一检索管线 | 已实现。对话、批量回放、评测共用 `hybrid_retrieve`。 |
| FR-2.6 | 检索诊断 | 已实现。记录 vector、keyword、merged、final context 阶段。 |
| FR-2.7 | Retrieval-only | 已实现。可跳过 LLM；检索模式独立支持 hybrid / vector / keyword，降低生成阶段波动对召回评估的影响。 |

### FR-3 对话与实验记录

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-3.1 | 流式问答 | 已实现。Ollama 流式输出。 |
| FR-3.2 | 无检索结果拒答 | 已实现。无上下文时不调用 LLM。 |
| FR-3.3 | 参考来源展示 | 已实现。展示文件、chunk、得分和片段。 |
| FR-3.4 | 检索诊断展示 | 已实现。展示候选阶段、最终上下文、重排变化等。 |
| FR-3.5 | 多会话 | 已实现。SQLite `chat_sessions` + `qa_log.session_id`。 |
| FR-3.6 | 人工评分 | 已实现。对当前会话最近一次 QA 打分和备注。 |
| FR-3.7 | QA 导出 | 已实现。支持 JSON / CSV。 |
| FR-3.8 | 批量回放 | 已实现。每行一个问题，结果落库为新会话。 |

### FR-4 评测平台

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-4.1 | 评测集导入 | 已实现。支持 JSON / CSV / Excel；同名导入会生成新版本，不覆盖历史 RUN。 |
| FR-4.2 | 评测 case 字段 | 已实现。支持 question、expected_answer、expected_file_names、expected_chunk_content、keywords、allow_abstain、tags、note。 |
| FR-4.3 | 完整 LLM 评测 | 已实现。执行检索、生成、命中判断和结果保存。 |
| FR-4.4 | Retrieval-only 评测 | 已实现。跳过 LLM；answer/abstain 不计分，只评估召回与上下文命中。 |
| FR-4.5 | 命中率指标 | 已实现。candidate/context/chunk + all-labeled 分母；Full LLM 才汇总 answer/abstain。 |
| FR-4.6 | 错误类型 | 已实现。包含 expected file unresolved、target file miss、chunk miss、rerank drop、abstain miss 等。 |
| FR-4.7 | 错误归因 | 已实现。基于 diagnostics 和 error_type 生成 attribution。 |
| FR-4.8 | 按 tag / file / error 汇总 | 已实现。评测面板展示多维汇总。 |
| FR-4.9 | RUN 参数快照 | 已实现。切片以索引 manifest 为准；记录模型、检索、query anchoring、降级标记等。 |
| FR-4.10 | Run fingerprint | 已实现。含 generation_mode / query_anchoring / index 切片，用于判断实验口径是否变化。 |
| FR-4.11 | RUN diff | 已实现。比较最近两次 RUN 的指标、参数、索引、数据质量和归因变化。 |
| FR-4.12 | Baseline 对比 | 已实现。可将当前评测集最新 RUN 设为 baseline，后续展示最新 RUN vs baseline。 |
| FR-4.13 | 参数网格 | 已实现。支持 retrieval-only 下的 keyword / vector / hybrid、Top-N / Top-K、rerank 组合网格，并按命中表现排序。 |
| FR-4.14 | 评测报告导出 | 已实现。导出 JSON，并生成 Markdown 报告。 |

### FR-5 评测治理

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-5.1 | 评测集质量扫描 | 已实现。识别空问题、缺 expected file、expected file unresolved、缺 chunk、缺答案、拒答口径冲突、缺 tag。 |
| FR-5.2 | Expected file alias 治理 | 已实现。展示 unresolved raw、候选文件、相似度原因，支持保存/删除 alias。 |
| FR-5.3 | Alias 持久化 | 已实现。保存到 `data/eval_file_aliases.json`。 |
| FR-5.4 | Alias 参与评测 | 已实现。后续解析 expected file 时自动使用 alias。 |
| FR-5.5 | Chunk 命中诊断 | 已实现。记录 expected chunk 与候选/context chunk 的相似度、覆盖率、公共字符数、最佳阶段和预览。 |
| FR-5.6 | Case 级检索漏斗 | 已实现。展示 vector / keyword / merged / final ranked / context 各阶段候选数和目标文件命中状态。 |

### FR-6 平台运维与稳定性

| 编号 | 需求 | 当前实现 |
| --- | --- | --- |
| FR-6.1 | Ollama 健康检查 | 已实现。检查本地 Ollama 基本状态和模型可见性。 |
| FR-6.2 | Chroma telemetry 噪声过滤 | 已实现。关闭匿名遥测并过滤 `Failed to send telemetry event` 噪声。 |
| FR-6.3 | SQLite 数据持久化 | 已实现。QA、session、eval dataset、eval run、case result、manifest 均落 SQLite。 |
| FR-6.4 | 不破坏用户数据 | 已实现为设计原则。索引清理仅作用于旧版本或 building 残留，不删除上传文件和评测数据。 |

## 4. 数据模型

### SQLite

主要表：

- `qa_log`
- `upload_log`
- `chat_sessions`
- `index_manifest`
- `eval_datasets`
- `eval_cases`
- `eval_runs`
- `eval_case_results`

### JSON 文件

- `data/ui_preferences.json`：UI 偏好。
- `data/eval_file_aliases.json`：expected file alias。
- `data/eval_baselines.json`：dataset -> baseline RUN。
- `data/exports/*.json` / `*.md`：评测报告。

### Chroma

Chroma 索引使用版本化目录管理。当前 active 版本由 SQLite `index_manifest.active_chroma_subdir` 指向；版本根目录为 `{chroma_dir.name}.__versions__`。

Windows 下若项目路径含非 ASCII 字符，`chroma_dir` 会自动重定向到 `%LOCALAPPDATA%\ClaudeCode\rag_lite_chroma\<project_slug>\chroma`，UI 会提示一次。

## 5. 当前验收口径

### 基础自检

```powershell
.\.venv\Scripts\python.exe -B self_test.py --import-all --no-gradio-import --skip-ollama
```

### UI 自检

- `main.build_ui()` 能正常返回 Gradio Blocks。
- 本地 HTTP 启动返回 200。
- 知识库、对话、评测主要按钮响应正常。

### 检索评测自检

- 可运行 retrieval-only 回归。
- 生成 RUN。
- 写入 `eval_case_results`。
- 显示 summary、结果明细、错误归因、chunk 诊断、检索漏斗。

### 数据治理自检

- Expected file alias 可保存、删除、刷新。
- Alias 保存后影响 expected file 解析。
- Baseline 可设置并用于对比。

## 6. 当前已知问题

1. 评测集口径仍是主要瓶颈。若 expected file 未解析，指标会被数据质量问题主导。
2. `allow_abstain=true` 同时带目标文件或目标片段时，需要人工判断是刻意测试还是标注冲突。
3. `chunk_hit_rate` 低时，需要结合 chunk 诊断判断是切片、Top-K、召回还是 expected chunk 标注问题。
4. 本地 Ollama 资源可能波动，完整 LLM 评测可能慢或失败；召回评估建议优先使用 retrieval-only。
5. `main.py` 仍较大；已拆出 `ui_kb.py`、`eval_judge.py`、`eval_platform.py`、`eval_runner.py`、`platform_ops.py` 等，后续维护应继续拆分对话/评测 UI。

## 7. Backlog

已落地（勿再当缺口）：

- Expected file alias 持久化与评测参与（`data/eval_file_aliases.json`）。
- Baseline 设置与最新 RUN vs baseline 对比（`data/eval_baselines.json`）。
- Retrieval-only 参数网格，并按 context / chunk / candidate 命中排序。
- 知识库索引运维诊断与旧版本清理预览（`ui_kb.py` / `platform_ops.py`）。

优先级建议：

### P0

- 对业务评测集补齐正式 alias，降低 `EXPECTED_FILE_UNRESOLVED`。
- 拆分拒答样本口径，处理 `ABSTAIN_WITH_TARGET_CONTEXT`。
- 基于治理后的干净 RUN 设置 baseline。

### P1

- 参数网格增加最佳配置推荐文案（排序已实现）。
- Chunk 诊断增加自动失败解释。
- Eval RUN 报告增加 baseline delta 和 top failure case。
- 增加更细的 per-case 导出。

### P2

- 继续拆分 `main.py`：
  - `rag_lite/ui_eval.py`
  - `rag_lite/eval_governance.py`
  - `rag_lite/eval_experiments.py`
  - `rag_lite/ui_chat.py`
- 为 alias、chunk diagnostics、retrieval funnel、baseline compare、param grid 增加单元测试。
- 增加文档截图或最小使用教程。
