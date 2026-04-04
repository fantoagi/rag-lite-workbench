"""Fail fast before pip: Python 3.14+ often has no Pillow/torch Windows wheels."""
from __future__ import annotations

import sys


def main() -> int:
    v = sys.version_info
    if v >= (3, 14):
        print(
            "[RAG-Lite] ��ǰ Python Ϊ {}.{}�����£�Pillow��torch ���� Windows �ϳ���Ԥ�������pip �᳢��Դ����벢�ױ�ȱ zlib �ȴ���\n"
            "�밲װ Python 3.12.x���Ƽ����� 3.11����װʱ��ѡ Add to PATH��Ȼ��ɾ�� ragZone �µ� .venv �ļ��У����������� start.bat / run.ps1��".format(
                v.major, v.minor
            ),
            file=sys.stderr,
        )
        return 1
    if v >= (3, 13):
        print(
            "[RAG-Lite] ��ʾ��Python 3.13 �ϲ�����������û�� Windows wheel���� pip ��������� 3.12��",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
