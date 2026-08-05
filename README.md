# RAG-Lite 沙箱验证平台

RAG-Lite 是一个面向业务、产品和算法同学的本地化 RAG 验证工作台。它把知识库构建、混合检索、对话回放、评测集运行、错误归因、参数对比和索引运维放在同一个 Gradio 应用里，目标不是做一个聊天 Demo，而是帮助团队稳定验证“哪些语料、切片、检索参数和模型组合更可靠”。

当前技术栈：

- UI：Gradio 5.50
- 编排：LlamaIndex
- 向量库：本地 Chroma，版本化索引目录
- 模型：Ollama 本地 LLM / Embedding
- 重排：可选 sentence-transformers CrossEncoder
- 存储：SQLite + JSON 配置文件
- 文件：PDF / TXT / Markdown / DOCX，PDF/DOCX 可选 OCR 与视觉增强

产品需求和实现口径见 [prd.md](prd.md)。

## 快速启动

首次使用前安装 [Ollama](https://ollama.com)，并准备至少一个 LLM 和一个 embedding 模型，例如：

```powershell
ollama pull qwen2.5:0.5b
ollama pull bge-m3
```

Windows 推荐直接双击：

- [start.bat](start.bat)
- 或 [在终端里运行我.cmd](<在终端里运行我.cmd>)

也可以手动运行：

```powershell
cd ragZone
powershell -ExecutionPolicy Bypass -File run.ps1
```

脚本会按 **3.12 → 3.11 → 3.10** 探测可用解释器并创建 `.venv`，安装依赖后启动 `main.py`。

Python 口径：

- **推荐**：3.12
- **支持**：3.11；脚本也接受 3.10
- **慎用**：3.13（部分依赖可能缺 Windows wheel，失败时请改用 3.12）
- **不支持**：3.14+

只创建环境、不启动 UI：

```powershell
powershell -ExecutionPolicy Bypass -File setup-venv.ps1
```

命令行启动：

```powershell
.\.venv\Scripts\python.exe -u main.py
```

默认访问地址为 `http://127.0.0.1:7860`，端口配置见 [config.yaml](config.yaml)。

## 自检

```powershell
.\.venv\Scripts\python.exe -u self_test.py
.\.venv\Scripts\python.exe -u self_test.py --import-all
.\.venv\Scripts\python.exe -u self_test.py --import-all --no-gradio-import --skip-ollama
```

常用轻量检查：

```powershell
.\.venv\Scripts\python.exe -B self_test.py --import-all --no-gradio-import --skip-ollama
```

单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 界面结构

应用只有三个顶层页签：

1. **知识库**：上传、切片、索引构建与运维
2. **对话**：单轮/多会话 RAG 问答、评分与导出
3. **评测**：评测集导入与运行、结果诊断、alias 治理；其下「实验工具」折叠区包含批量回放、RUN 对比、baseline、参数网格、Ollama 健康检查和报告导出

## 使用流程

### 1. 知识库

在“知识库”页完成文档上传和索引构建：

1. 选择嵌入模型、切分策略、chunk size、overlap。
2. 上传 PDF / TXT / Markdown / DOCX。
3. 保存到上传目录。
4. 构建向量索引。
5. 查看文件状态、切片预览、切片统计和索引运维诊断。

切分策略支持 `sentence` / `token` / `paragraph`。界面默认选中 **paragraph**（段落优先）；代码在未传入策略时回退为 `sentence`。

索引构建采用“临时目录构建 -> 自检 -> 版本化激活”的方式。新版本通过自检后才会切换到 `{chroma_dir.name}.__versions__/<build_id>`，失败时保留上一个可用版本。

知识库页还提供：

- 上传文件状态：新增、已变更、已入库、模型不一致等。
- 切片预览：按文件查看 Chroma 中的实际 chunk。
- 切片统计：文件数、chunk 数、平均字符数、空块、OCR/视觉提示块。
- 索引运维：真实活跃 Chroma 目录、manifest 指向、版本数量、building 残留、磁盘占用。
- 清理预览：可 dry-run 清理旧索引版本和残留 building 目录，默认保留 active 和最近历史版本。

### 2. 对话

在“对话”页验证单轮 RAG 问答：

1. 选择 LLM、Top-N、Top-K、是否 rerank、LLM `num_ctx`。
2. 输入问题。
3. 查看参考来源和检索诊断。
4. 对回答打分或备注。
5. 导出 QA 日志 JSON / CSV。

对话固定走 hybrid 检索。检索诊断会展示 vector、keyword、merged、final context 等阶段的候选数量、来源、得分和片段摘要。若没有检索到任何上下文，系统不会调用 LLM，会直接给出固定拒答说明。向量阶段失败时会降级为 `keyword_fallback` / `vector_error`，不再伪标为 hybrid。

### 3. 批量回放

在“评测”页的「实验工具」中，批量回放支持“每行一个问题”，会自动创建一个新会话并保存：

- 问题
- 答案
- 来源片段
- 参数快照
- 检索诊断

适合快速验证一组业务问题在当前索引和参数下的表现。回放可选择 hybrid / vector / keyword。

### 4. 评测平台

评测功能是当前版本的核心能力。支持导入评测集并运行完整评测或 retrieval-only 评测。

评测样本字段包括：

- question
- expected_answer
- expected_file_names
- expected_chunk_content
- expected_answer_keywords
- allow_abstain
- tags
- note

评测运行会生成 RUN，并记录：

- candidate / context / chunk hit rate（另有 all-labeled 口径：未解析 expected file 计入分母为未命中）
- answer hit rate / abstain accuracy（仅 Full LLM；retrieval-only 显示为 —）
- error type / attribution
- run fingerprint（含 generation_mode、query_anchoring、index 切片）
- 参数快照与检索诊断 JSON

当前评测面板包括：

- 总览：最新 RUN、指标、参数。
- 结果明细：逐 case 命中状态、错误类型、归因、QA ID。
- 诊断分析：错误类型、tag 汇总、file 汇总。
- chunk 诊断：expected chunk 与候选/context chunk 的相似度、覆盖率、公共字符数和最佳片段。
- 检索漏斗：vector / keyword / merged / final ranked / context 各阶段候选数和目标文件命中状态。
- 评测治理：评测集质量问题、失败样本、expected file alias 治理。

Query anchoring（把 expected 文件名拼进检索问题）**默认关闭**。真实召回 / baseline / 参数网格请保持关闭；仅在做“给定目标文档名”的对照实验时再打开。

### 5. Expected File Alias 治理

评测集中常见 `《投资学》`、`《XXX公司年报》` 这类业务名，而本地文件名通常是完整 PDF 文件名。Alias 治理用于建立二者映射。

在评测页“Expected file alias 治理”中可以：

- 查看 unresolved expected file。
- 查看候选目标文件和相似度原因。
- 点击行自动回填 raw 和 target。
- 保存 alias。
- 删除 alias。

Alias 保存到：

```text
data/eval_file_aliases.json
```

保存后会参与后续评测、回归和治理面板刷新。

同名评测集再次导入时会自动生成带时间戳的新版本，不会删除已有 RUN、baseline 或 diff 历史。

### 6. 实验对比

在评测页「实验工具」中支持：

- QA 实验聚合对比。
- Eval RUN 对比。
- 最近两次 RUN 差异解释。
- Baseline 设置与最新 RUN vs baseline 对比。
- Retrieval-only 参数网格，可显式比较 keyword / vector / hybrid 和 rerank 组合，并按命中表现排序。
- Ollama 健康检查。
- 最新评测报告导出。

Retrieval-only 参数网格会跳过 LLM 生成，适合在本地 Ollama 资源紧张时快速比较 keyword / vector / hybrid 以及 Top-N / Top-K / rerank 对召回和上下文命中的影响。

## 数据目录

默认数据目录：

```text
data/
```

关键文件和目录：

- `data/uploads/`：上传文件。
- `data/rag_lite.db`：SQLite 数据库。
- `data/exports/`：导出的 QA 或评测报告。
- `data/ui_preferences.json`：UI 选择偏好。
- `data/eval_file_aliases.json`：评测 expected file alias。
- `data/eval_baselines.json`：评测 baseline RUN。
- `{chroma_dir.name}.__versions__/`：版本化 Chroma 索引目录。

Windows 下若项目路径含非 ASCII 字符（例如中文目录），Chroma 会自动重定向到：

```text
%LOCALAPPDATA%\ClaudeCode\rag_lite_chroma\<project_slug>\chroma
```

知识库页会提示一次重定向信息；真实活跃目录以索引运维诊断为准，不要只看 `data/chroma`。

## 重要实现口径

- 默认使用混合召回：向量召回 + 本地关键词/BM25 召回，合并去重后可选 rerank。
- 对话、批量回放、评测共用同一检索管线（`engine.hybrid_retrieve`）。
- Retrieval-only 评测会跳过 LLM，且不把 answer/abstain 计入汇总；检索模式可单独选择 hybrid / vector / keyword。
- 空检索在评测中写入 `[NO_CONTEXT]` 哨兵，优先归因 `TARGET_FILE_MISS`，避免误标为模型拒答。
- RUN 切片参数以索引 manifest 为准；UI 切片控件仅作对照，不一致时会告警。
- RUN fingerprint 记录 generation_mode、query_anchoring、检索/索引快照等，用于区分实验口径。
- Chroma telemetry 噪声已被过滤，评测日志只保留业务相关输出。
- 不做部署治理：当前版本不包含权限、审计、脱敏导出、模型版本强绑定等合规能力。

## 当前建议工作流

1. 上传并构建知识库。
2. 刷新索引诊断，确认 active Chroma 与 manifest 一致。
3. 导入评测集。
4. 先运行 retrieval-only 回归。
5. 在 alias 治理中处理 unresolved expected file。
6. 再运行 retrieval-only 回归并设为 baseline。
7. 跑参数网格，查看候选命中、上下文命中、片段命中变化。
8. 对少量关键 case 跑完整 LLM 评测。
9. 导出评测报告。

## 已知限制

- 评测效果强依赖评测集标注质量，尤其是 expected file 和 expected chunk。
- `allow_abstain=true` 同时带目标文件/目标片段时，口径需要人工确认。
- 本地 Ollama 资源紧张时，LLM 或 embedding 可能启动慢或失败；建议优先使用 retrieval-only 做召回评估。
- OCR/视觉增强会显著增加构建耗时，默认建议关闭，只在专项验证时开启。
- `main.py` 仍承担较多 UI 与评测编排逻辑；已拆出 `ui_kb.py`、`eval_*`、`platform_ops.py` 等模块，后续可继续拆分对话/评测 UI。
