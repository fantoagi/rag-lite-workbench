"""Fail fast before pip: Python 3.14+ often has no Pillow/torch Windows wheels."""
from __future__ import annotations

import sys


def main() -> int:
    v = sys.version_info
    if v >= (3, 14):
        print(
            "[RAG-Lite] 当前 Python 为 {}.{}, 过高：Pillow、torch 等在 Windows 上缺少预编译包，"
            "pip 会尝试源码编译并容易缺 zlib 等依赖。\n"
            "请安装 Python 3.12.x（推荐）或 3.11；安装时勾选 Add to PATH，"
            "然后删除 ragZone 下的 .venv 文件夹，再重新运行 start.bat / run.ps1。".format(
                v.major, v.minor
            ),
            file=sys.stderr,
        )
        return 1
    if v >= (3, 13):
        print(
            "[RAG-Lite] 提示：Python 3.13 上部分依赖可能没有 Windows wheel；若 pip 失败请改用 3.12。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
