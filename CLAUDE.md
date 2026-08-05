# ragZone

## Working rules
- Preferred startup: `start.bat` or `powershell -ExecutionPolicy Bypass -File run.ps1`.
- Do not use system `python` for this project when running the app, installing deps, or debugging environment issues.
- Use the project interpreter: `./.venv/Scripts/python.exe` on Windows.
- Before running any Python command for ragZone, verify `sys.executable` points to `ragZone/.venv`; if not, stop instead of continuing with debugging, builds, or evaluation.
- Python policy: prefer 3.12; support 3.11; scripts also accept 3.10; warn on 3.13; block 3.14+. If 3.13 has wheel/install issues, switch to 3.12.
- Install dependencies into `ragZone/.venv` only.
- Before changing dependency or environment setup, check `README.md`, `last-run.log`, and `last-run-error.txt`.
- If `.venv` appears broken or was created with an unsupported Python, prefer recreating it via `start.bat` or `setup-venv.ps1` instead of ad-hoc fixes.

## Notes
- This project already contains scripts that auto-select a supported Python and repair the virtual environment.
- Keep `CLAUDE.md` minimal here; detailed setup and troubleshooting belong in `README.md`.
