# RAG-Lite 沙箱验证平台（MVP）

对应 [`prd.md`](prd.md)：本地 Chroma 向量库 + Ollama（LLM / Embedding，**界面可选并记住上次选择**）+ 可选应用内 Cross-Encoder 重排 + Gradio UI；**知识库**侧管理嵌入模型与切分策略，**对话**侧管理 LLM 与会话历史。

## 一键启动（Windows）

1. **首次使用前**：安装 [Ollama](https://ollama.com)，并按下文拉取模型（仅需一次）。
2. **双击 [`start.bat`](start.bat)**（或 [`在终端里运行我.cmd`](在终端里运行我.cmd)，或 `powershell -ExecutionPolicy Bypass -File run.ps1`）。窗口会**在结束后等待按键**再关闭；完整输出在 **`ragZone/last-run.log`**，异常摘要（若有）在 **`last-run-error.txt`**。脚本会**自动**：优先用 **`py -3.12` / `-3.11` / `-3.10`** 创建 `.venv`（避免默认落到 3.14）；若检测到现有 `.venv` 是 **Python 3.14+** 会**自动删掉再建**；再自动补 pip、装依赖、启动 `main.py`。若本机没有可用的 3.10–3.13，会**尝试 `winget install Python.Python.3.12`**（需已启用 winget，必要时 UAC）。
3. 若只想先建好环境不启动 UI：`powershell -ExecutionPolicy Bypass -File setup-venv.ps1`（同样自动选 Python / 重建坏 venv）。
4. **Cursor / VS Code**：工作区若打开上层「智能体」文件夹，已指向 `ragZone/.venv`；若**单独打开 `ragZone` 文件夹**，请使用 [`ragZone/.vscode/settings.json`](.vscode/settings.json) 中的解释器路径。调试配置见上层 `.vscode/launch.json`。

无需再手动 `set PYTHONPATH`：`main.py` 已把项目根目录加入 `sys.path`。

## 环境

1. 安装 [Ollama](https://ollama.com)，并拉取所需模型。`config.yaml` 中的 **`ollama.llm_model` / `ollama.embed_model`** 为**默认**；启动后可在界面下拉框选择本机已安装模型（来自 `GET /api/tags`，并与配置里可选列表合并），选择会保存到 **`data/ui_preferences.json`**，下次启动仍生效。

   默认配置为小体量 LLM（省显存），例如：

   ```bash
   ollama pull qwen2.5:0.5b
   ollama pull bge-m3
   ```

   需要略好效果但仍较轻时，可改用 `qwen2.5:1.5b` 等，在界面选择或在 `config.yaml` 里改默认 `llm_model`。可在 `config.yaml` 的 `ollama` 下配置可选列表 **`llm_models`** / **`embed_models`**（见文件内注释），便于预置多候选做对比。

   **关于 [Qwen3.5-0.8B（ModelScope）](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B)**：该权重在 Ollama 官方库里通常**没有**与网页完全同名的 `pull` 标签；若要坚持用这一 checkpoint，需要自行下载 **GGUF**（或能从社区找到对应 Ollama 模型）后，用官方文档中的 **`ollama create`** 导入为本地模型名，再在界面或 `config.yaml` 中选用该名称。

   若本地无 `bge-m3`，请在界面嵌入模型下拉框中选用已安装的嵌入模型（如 `nomic-embed-text`），或改 `config.yaml` 默认 `embed_model`。

2. **Gradio 版本**：`main.py` 与依赖栈以 **`gradio==5.50.0`**（及 `gradio-client==1.14.0` 等）为锁定目标，与 `huggingface-hub` 1.x / `transformers` 5.x 兼容。若环境里误装了 **Gradio 6+**，`main.py` 会尝试对齐回 5.50。若不想自动执行 pip，可设 **`RAG_LITE_SKIP_AUTO_PIP=1`**。

4. **Python 请用 3.12 或 3.11（Windows 最省心）**；**不要使用 3.14**（Gradio 依赖的 Pillow 等尚无可靠预编译包，pip 会源码编译并常见 **`zlib` / `Failed building wheel for pillow`**）。3.13 若 pip 失败也请改 3.12。**依赖请安装到 `ragZone/.venv`**（`run.ps1` / `start.bat` / `setup-venv.ps1` 会代劳）。手动安装示例：

   ```powershell
   cd ragZone
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   若提示无法运行脚本，可先执行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

## 启动（命令行方式）

```bash
cd ragZone
.\.venv\Scripts\python.exe -u main.py
```

浏览器访问 `http://127.0.0.1:7860`（端口与是否自动打开浏览器见 `config.yaml` 的 `ui` 段）。

## 自检（不启动 Web）

在 `ragZone` 目录执行（建议加 `-u` 无缓冲，便于立刻看到输出）：

```bash
python -u self_test.py
python -u self_test.py --import-all
python -u self_test.py --no-gradio-import --skip-ollama
```

`--no-gradio-import` 可在已确认是 Gradio 6、import 极慢时跳过耗时段；其余仍会检查配置与数据库目录。

## 使用顺序

1. **知识库**：可选 **刷新 Ollama 模型列表** → 选择 **嵌入模型** → 选择 **切分策略**（按句 / Token / 段落优先）与 Chunk / Overlap → 上传 PDF/TXT/Markdown/DOCX →「保存到上传目录」→「构建向量索引」。下方 **表格** 显示各文件大小与相对上次成功构建的**索引状态**；可从上传目录 **移除** 指定文件（删除后若需同步向量请重新构建）。
2. **对话**：选择 **对话模型（LLM）** → 在 **历史会话** 中切换或「新建会话」→ 设置 Top-N / Top-K、是否重排、系统提示词 → 输入问题并发送；右侧/下方为 **参考来源**（片段与得分）。
3. **人工评估**：对**最近一次**回答打分并保存（仍针对当前会话内最后一次入库的问答）。
4. **实验导出**（可选）：导出 **全库** 问答日志为 JSON/CSV（含 `session_id`）；日常查看历史以「对话」页 **会话列表** 为主。

数据目录默认：`ragZone/data/`（`uploads/`、`chroma/`、`rag_lite.db`、`exports/`、**`ui_preferences.json`**）。

## 故障排除

- **`Failed building wheel for pillow` / `could not be found for zlib`**：几乎总是 **Python 太新（如 3.14）**。请先安装 **Python 3.12**（可与 3.14 共存）；再运行 **`start.bat` / `run.ps1`**：会自动优先使用 `py -3.12` 并**删除不兼容的 `.venv` 再重建**，无需手工删文件夹。若本机只有 3.14、已装 **winget**，脚本也会尝试自动安装 3.12。

- **`No module named pip` / 卡在 `ensurepip` 很久**：说明 `.venv` 里没有 pip，或 `ensurepip` 在你当前路径/杀毒环境下极慢。**请先关闭卡住的窗口**，再重新运行 `start.bat` / `run.ps1`：现会**优先**用系统自带的 `pip install ... --target .venv\Lib\site-packages`（一般有下载进度），失败才 `ensurepip`，再失败才下载 `get-pip.py`。手动修复示例：  
  `py -3 -m pip install pip setuptools wheel --target "完整路径\ragZone\.venv\Lib\site-packages"`  
  然后 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`。

- **卡在「创建虚拟环境 / python -m venv」很久或不动**：`start.bat` 已改为**优先 `py -3 -m venv`**（减少 Windows 商店版 `python` 干扰），并在成功后打印 **`.venv 就绪`**。若仍超过约 **5 分钟**：① 任务管理器看是否有 `python.exe` 占用 CPU；② **中文或过长路径**下 venv 偶发极慢，可将 `ragZone` 复制到短英文路径（如 `D:\work\ragZone`）再建 `.venv`；③ 杀毒对新建大量小文件实时扫描会拖慢，可暂时排除 `ragZone\.venv`；④ 改在 PowerShell 执行 **`.\setup-venv.ps1`** 看是否有报错输出。

- **卡在「正在 pip install…」很久**：常见是下载 `torch` 或依赖解析慢；`main.py` 会**优先只对齐 Gradio 5.50 栈**（比整包重装通常快）。请用 **`python -u main.py`** 以便尽快看到 pip 输出。也可先手动执行  
  `python -m pip install gradio-client==1.14.0 gradio==5.50.0 "tomlkit>=0.12.0,<0.14.0" --force-reinstall`  
  再启动。

- **点右上角「Run Python File」仍像没反应**：Cursor/VS Code 有时不把输出送到当前终端，或未安装 **Python / Debugpy 扩展**。请改用下面任一方式（任选一种即可）：
  1. **菜单 `终端` → `运行任务…` → 选 `RAG-Lite: 终端运行（工作区含 ragZone）`**（或「工作区即 ragZone」，取决于你打开的是哪一层文件夹）；会在**专用终端**里执行 `python -u main.py`。
  2. **运行和调试**：侧栏播放键 → 选择 **「RAG-Lite: main（工作区含 ragZone）」** → 开始调试。需已安装扩展：`ms-python.python`、`ms-python.debugpy`（见 [.vscode/extensions.json](../.vscode/extensions.json)）。
  3. 在资源管理器中双击 [`在终端里运行我.cmd`](在终端里运行我.cmd)，或在 CMD 里 `cd` 到 `ragZone` 后执行：`python -u main.py`。
  4. 若卡在 **`正在 import Gradio…` 后几分钟**：多为机械硬盘或杀毒软件（如 Defender）在扫描 `torch` 等 DLL，属正常现象；可尝试将 Python 环境目录加入排除、换 SSD，或改用较轻的 Python 环境（仅本项目的 venv）。**Gradio 加载完成后**才会打印 `Gradio 已就绪，本步耗时 … 秒`；之后的 Chroma/检索模块改为**首次点「构建索引」或「发送」**时再加载，并会另有提示。

  5. **`import gradio` / `HfFolder` / `huggingface_hub`**：**Gradio 4.44** 依赖已移除的 `HfFolder`，而 **`transformers` 5.x 需要 `huggingface-hub` 1.x**，两者无法兼得。本项目已固定 **`gradio==5.50.0`** + **`gradio-client==1.14.0`**（与 hub 1.x 兼容）。若仍报错，删除 **`ragZone/.venv`** 后重跑 **`start.bat`**。
  6. **其它 Gradio 堆栈**（`tomlkit` / Pydantic）：与 **Gradio 5.50** 元数据一致（如 `pydantic>=2.0,<=2.12.3`）；`main.py` 会在异常时自动重装。

- **`No module named 'gradio'`（点 IDE 运行按钮）**：`main.py` 会在导入前检测 `gradio`；若缺失会**自动用当前解释器**执行 `pip install -r requirements.txt`。首次可能较慢，终端里应看到 `[RAG-Lite] 正在安装依赖…`。若仍失败，检查网络/权限，或在该解释器下手动：`python -m pip install -r requirements.txt`。

- **`ollama pull` 报错 `TLS handshake timeout` / 连接 `registry.ollama.ai` 失败**：这是访问 Ollama 官方寄存器时的**网络或 TLS 问题**（国际链路、公司防火墙、DNS 等），不是模型名称错误。可依次尝试：
  1. **换网络或时段重试**；或使用可访问外网的 **VPN**。
  2. **走 HTTP 代理**（若本机有 Clash、V2Ray 等，端口以你实际为准，常见为 `7890`）。在**同一窗口**先设环境变量再 `pull`（PowerShell 示例）：
     ```powershell
     $env:HTTPS_PROXY="http://127.0.0.1:7890"
     $env:HTTP_PROXY="http://127.0.0.1:7890"
     ollama pull qwen2.5:0.5b
     ```
     CMD 示例：`set HTTPS_PROXY=http://127.0.0.1:7890` 与 `set HTTP_PROXY=...` 后执行 `ollama pull`。
  3. **不从官方 pull**：在 [ModelScope](https://modelscope.cn) / Hugging Face 下载对应模型的 **GGUF**，按 [Ollama 文档](https://github.com/ollama/ollama) 用 `Modelfile` 执行 `ollama create`，再在 `config.yaml` 里把 `llm_model` / `embed_model` 改成你本地的模型名。
