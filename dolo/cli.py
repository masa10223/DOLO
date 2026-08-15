"""DOLO の統合コマンドライン入口。"""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dolo", description="Drosophila pose tracking and analysis"
    )
    parser.add_argument("--version", action="store_true", help="バージョンを表示")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="ブラウザGUIを起動")
    gui.add_argument("--host", default="127.0.0.1", help="bind address（既定: 127.0.0.1）")
    gui.add_argument("--port", type=int, default=8080, help="port（既定: 8080）")
    gui.add_argument("--model", help="defaultモデル重みのパス")
    gui.add_argument("--data-dir", help="アップロードと結果の保存先")
    gui.add_argument("--reload", action="store_true", help="開発用auto reload")
    gui.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")

    doctor = sub.add_parser("doctor", help="環境とdefaultモデルを診断")
    doctor.add_argument("--model", help="確認するモデル重みのパス")
    doctor.add_argument("--data-dir", help="確認するデータ保存先")
    return parser


def main(argv: list[str] | None = None) -> int:
    from dolo import __version__

    parser = _parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "doctor":
        from dolo.gui.runtime import doctor_report

        report, ready = doctor_report(args.model, args.data_dir)
        print(report)
        return 0 if ready else 1
    if args.command == "gui":
        from dolo.gui.app import run_gui

        run_gui(
            host=args.host,
            port=args.port,
            model=args.model,
            data_dir=args.data_dir,
            reload=args.reload,
            show=not args.no_open,
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
