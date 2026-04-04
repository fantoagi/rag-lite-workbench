# 第一部分：RAG 本地轻量化验证系统 PRD（产品需求文档）

**文档说明**：本节描述产品目标与需求；**「当前实现」**以 `ragZone/` 仓库代码为准（截至本文更新）。**尚未落地项**统一记在文末 **「待办事项」**，避免与交付预期混淆。

---

### 1. 项目概述与定位

* **项目名称**：RAG-Lite 沙箱验证平台 (MVP版)
* **项目定位**：一款开箱即用、完全本地化部署的轻量级 RAG 测试环境。
* **核心目标**：帮助非开发人员（业务专家、产品经理）与算法工程师共同**验证业务私有数据在 RAG 系统下的表现质量**。无需搭建复杂的微服务集群，即可快速测试不同文档切片策略、不同检索模型和生成模型对最终问答准确率的影响。

### 2. 核心业务流程 (User Flow)

1. **语料喂入（离线）**：业务人员上传业务文档 → 在「知识库」选择 **嵌入模型（Ollama）**、**切分策略**与 Chunk / Overlap → 点击构建索引 → 系统将切块向量化并写入本地 Chroma；界面表格展示各文件是否已与最近一次成功构建一致，支持从上传目录移除文件（详见 FR-1）。
2. **问答验证（在线）**：用户在「对话」选择 **对话模型（LLM）** → 选择或新建 **会话** → 在聊天框输入问题 → 系统先向量（及可选重排）检索 → **有命中片段时**流式生成答案，并在 **HTML「参考来源」区** 高亮展示引用片段与得分；**无命中片段时不调用大模型**，仅输出固定说明（避免无依据胡答）。用户可评分、备注；问答按会话落库，**导出 JSON/CSV** 为全库备份选项（见 FR-3、FR-4）。

---

### 3. 功能需求清单与当前实现

#### 模块一：知识库构建与管理 (Data Ingestion)

| 编号 | 需求摘要 | 当前实现 |
| :--- | :--- | :--- |
| FR-1.1 | 解析 PDF / TXT / Markdown / DOCX；单文件大小可配置上限 | **已实现**：Gradio 多文件上传到 `uploads`；`ingest.save_uploads` 校验后缀与 `max_file_size_mb`（默认 50MB，与 `config.yaml` 一致）。解析走 LlamaIndex `SimpleDirectoryReader`（电子文本型 PDF 为主；扫描件/复杂版式/密码 PDF 不保证）。 |
| FR-1.2 | 界面调节切分策略、Chunk Size、Overlap | **已实现**：「知识库」页 **切分策略**（`sentence` / `token` / `paragraph`）与 Slider；构建时分别对应 `SentenceSplitter`、`TokenTextSplitter`、段落优先的 `SentenceSplitter`（`paragraph_separator="\n\n"`）。参数快照含 `chunk_mode`。 |
| FR-1.3 | 向量索引本地落盘，无独立向量库服务端 | **已实现**：Chroma `PersistentClient` 写入 `data/chroma`（路径见 `config.yaml`）。 |
| FR-1.4 | 嵌入模型可选；与 Ollama 已安装模型对齐 | **已实现**：「知识库」嵌入模型下拉框，选项来自 `GET {ollama.base_url}/api/tags` 与 `config.yaml` 可选列表 `embed_models` / 默认 `embed_model` 合并；选择写入 `data/ui_preferences.json`。仅一个候选时控件为只读等效固定展示。 |
| FR-1.5 | 已上传文件与索引状态可见；支持从上传目录移除 | **已实现**：Markdown 表格展示文件名、大小、相对「上次成功构建」清单的状态（已入库 / 已变更需重建 / 新增未索引等）；若当前所选嵌入模型与上次构建不一致有提示。下拉选择文件后可「移除所选」删除磁盘文件；向量侧需重新构建以同步。成功构建后 SQLite **`index_manifest`** 记录嵌入模型、切分参数与各文件 mtime/size 快照。 |

#### 模块二：检索与重排 (Retrieval & Rerank)

**范围（未变）**：仅 **单向量召回 + 可选 Cross-Encoder 重排**；关键词/BM25/多路融合等不在本期交付内（见 **待办事项** 若后续要做）。

| 编号 | 需求摘要 | 当前实现 |
| :--- | :--- | :--- |
| FR-2.1 | Top-N 向量初筛，界面可调 | **已实现**：对话页 Slider；`engine.retrieve` 使用 `VectorIndexRetriever(similarity_top_k=top_n)`（并对 `from_vector_store` 空 `node_ids` 问题使用 `node_ids=None` 全库检索）。 |
| FR-2.2 | 重排开关；Cross-Encoder 对候选打分 | **已实现**：复选框；`rerank.rerank_nodes`（sentence-transformers `CrossEncoder`），关闭时取向量 Top-K 截断。 |
| FR-2.3 | Top-K 进入 Prompt，且 K ≤ N | **已实现**：界面约束 + `retrieve` 内 `top_k = min(top_k, top_n)`。 |

#### 模块三：对话、溯源与 Prompt (Generation & Tracing)

| 编号 | 需求摘要 | 当前实现 |
| :--- | :--- | :--- |
| FR-3.1 | 流式对话 | **已实现**：`Ollama.stream_chat` 流式回写到 Gradio Chatbot。 |
| FR-3.2 | 底部「参考来源」折叠区：文档名、Chunk、得分 | **已实现**：`gr.HTML` + `_sources_panel_html`：`<details>` 折叠、`<mark>` 高亮片段；向量相似度（Chroma 距离换算）或重排分数展示。 |
| FR-3.3 | 系统提示词可编辑 | **已实现**：对话页多行文本框，写入 `stream_answer` 的 SYSTEM 消息。 |
| FR-3.4 | 对话模型可选；便于切换对比 | **已实现**：「对话」页 LLM 下拉框，选项来源同 FR-1.4（`llm_models` / `llm_model`）；选择持久化到 `data/ui_preferences.json`。检索/生成分别使用界面选中的 `embed_model`（加载 Chroma 索引）与 `llm_model`（生成）。 |
| FR-3.5 | 多会话列表；点选查看历史问答 | **已实现**：SQLite **`chat_sessions`**；**`qa_log.session_id`** 关联会话。界面「历史会话」下拉切换会话内容；「新建会话」开启空对话。 |

#### 模块四：实验记录与导出

| 编号 | 需求摘要 | 当前实现 |
| :--- | :--- | :--- |
| FR-4.1 | 问答关联参数快照 | **已实现**：`engine.build_params_snapshot` 含 `chunk_mode` / chunk_size / chunk_overlap / top_n / top_k / rerank / `prompt.version` / `llm_model` / `embed_model` / `rerank_model`；随 `insert_qa` 写入 SQLite（含 **session_id**）。 |
| FR-4.2 | 人工评分与备注落库 | **已实现**：1–5 分 + 备注；`ExperimentStore.update_qa_rating`。 |
| FR-4.3 | 导出 JSON / CSV（全库备份） | **已实现**：「实验导出」页；`store.export_json` / `export_csv`（导出字段含 **session_id**）。日常回顾以「对话」页会话列表为主，导出为可选备份。 |

---

### 4. 非功能性需求 (NFR) 与现状

| 条目 | 说明 |
| :--- | :--- |
| 数据隐私 | **符合设计**：推理与检索本地；业务语料不出公网（依赖用户环境不把 Ollama/模型指向外网）。 |
| 部署轻量化 | **基本一致**：`start.bat` + 本地 venv；未强制 Docker/K8s/MySQL/Redis。README 含 Python 版本与依赖说明。 |
| 硬件与量化建议 | **文档级**：PRD 仍保留验收表述；具体「推荐配置表」以 README 为准，**精细化验收矩阵**见待办。 |
| 合规与审计扩展 | **未做**：操作审计、导出脱敏、语料/模型版本强绑定等见 **待办事项**。 |

---

### 5. 待办事项（Backlog）

以下条目在 PRD 中曾有描述或规划，**当前代码未覆盖或未完整覆盖**；实施时请拆任务并更新本文档。

1. **整目录 / 文件夹作为知识库导入**：现仅支持 Gradio 多文件选择，不支持「选一个本地文件夹」一键入库（FR-1.1 原文「单文件或文件夹」中的文件夹能力）。
2. **关键词检索 / BM25 / 多路召回与融合**：明确排除在 MVP 之外；若产品升级为「二期」，需单独 PRD 与接口设计。
3. **上传文件去重与版本**：SQLite `upload_log` 仅记录路径与大小；无文件哈希、无版本/去重策略。
4. **知识库一键清空与级联删除**：已支持**按文件**从上传目录移除并提示重建索引；**未提供**「清空整个上传目录 + 删 Chroma 集合 + 清空 `index_manifest`」的单按钮运维能力（若需要可二期封装）。
5. **检索失败 / 空库的运维提示增强**：已有基础文案与拒答逻辑；可选增加「Chroma 条数自检、embedding 连通性探测」等面向业务人员的诊断面板。
6. **合规与审计（NFR 扩展）**：操作日志、导出脱敏、强制语料与模型版本标识等。
7. **硬件推荐配置「验收表」**：按 GPU 显存 / CPU 场景整理成可勾选的发布检查表（与 README 联动）。

---

## 第二部分：轻量化系统架构设计 (Architecture Design)

### 1. 当前落地选型（与 PRD 原表的关系）

原「Gradio 或 Streamlit 二选一」：**已选定 Gradio（5.x）**，单栈维护。

其余分层保持不变，实装映射如下：

* **表现层**：Gradio — `main.py`（**知识库**：嵌入模型、切分策略、上传与索引状态、构建日志；**对话**：LLM、会话列表、问答与溯源；**实验导出**：全库 JSON/CSV）。
* **编排层**：LlamaIndex — `rag_lite/ingest.py`、`rag_lite/engine.py`。
* **检索增强**：可选 `rag_lite/rerank.py`（Cross-Encoder，不经 Ollama）。
* **持久化**：Chroma（块向量）+ SQLite（`qa_log`、`upload_log`、**`chat_sessions`**、**`index_manifest`**）— `rag_lite/store.py`；界面模型偏好 — `data/ui_preferences.json`（`rag_lite/prefs.py`）。
* **推理**：Ollama — LLM 与 Embedding；界面所选模型名可与 `config.yaml` 默认不同，以运行时下拉与 `ui_preferences` 为准。

### 2. 核心架构分层图（目标架构，仍适用）

```text
+--------------------------------------------------------------------+
|                       [表现层 (UI Layer)]                          |
|  功能：交互式Chat界面、知识库上传面板、策略参数调节表单、溯源展示        |
|  当前：Gradio                                                       |
+--------------------------------------------------------------------+
                                 | (HTTP / WebSocket 通信)
+--------------------------------------------------------------------+
|                    [应用编排层 (Orchestration)]                    |
|  组件1：文档加载器 (Document Loaders) - 负责解析多格式文件            |
|  组件2：文本切分器 — 按句 / Token / 段落优先（可配置）                |
|  组件3：检索管道 - 向量召回 + 可选 Cross-Encoder 重排               |
|  组件4：提示词组装 - 上下文 + 用户问题（无上下文时不调 LLM）          |
+--------------------------------------------------------------------+
              | (存取数据)                             | (本地推理调用)
+-----------------------------+       +------------------------------+
| [数据持久层 (Storage)]       |       | [模型推理层 (Inference)]       |
| 1. ChromaDB：块级向量+元数据 |       | A) Ollama：LLM、Embedding       |
| 2. SQLite：上传记录/QA/评分  |       | B) 应用内 Cross-Encoder（重排）   |
+-----------------------------+       +------------------------------+
```

### 3. 数据持久职责（单一真相源约定）

* **ChromaDB**：向量与可检索文本块、溯源展示以检索结果为准。
* **SQLite**：问答日志（含 **session_id**）、参数快照、评分备注、上传流水、**会话元数据**、**最近一次成功索引清单**；**不**冗余存储与 Chroma 完全一致的块正文。

### 4. 轻量化开源技术栈选型矩阵（参考）

| 架构层级 | 推荐开源工具选型 | 当前项目 |
| :--- | :--- | :--- |
| 表现层 | Gradio 或 Streamlit | **Gradio** |
| 编排层 | LlamaIndex | **LlamaIndex** |
| LLM + Embedding | Ollama | **Ollama** |
| 重排 | BGE-Reranker 等（应用内） | **CrossEncoder**（`config.yaml` 可配模型名） |
| 数据持久层 | ChromaDB + SQLite | **已实现** |

### 5. 核心机制设计说明

* **双段式检索**：Top-N 向量初筛 → 可选 Cross-Encoder 重排 → Top-K 拼上下文；N、K 界面可调。
* **本地化闭环**：本机回环访问 Ollama；重排依赖 PyTorch / sentence-transformers，与 Ollama 调用链分离。
* **无检索结果**：不向 LLM 发送可编造上下文；与系统提示词共同约束「拒答」行为（实现于 `main.py` + `engine.stream_answer`）。

---

（完）
