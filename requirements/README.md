# requirements/

依存関係の定義とロックファイル。詳しい手順は `docs/ENVIRONMENT.md` を参照。

| ファイル | 役割 |
|---|---|
| `core.in` | torch 不要のコア依存。これだけでユニットテストが通る |
| `gui.in` | NiceGUI。推論環境のlock後に追加でインストールする |
| `server.in` | GPU サーバー用の入力（`core.in` + torch 2.0.0 + 学習/解析） |
| `macos.in` | Mac 用の入力（`core.in` + torch 2.2.2 + 学習/解析） |
| `lock-linux.txt` | `server.in` を x86_64 Linux / Python 3.10 向けに解決した結果 |
| `lock-macos.txt` | `macos.in` を arm64 macOS / Python 3.10 向けに解決した結果 |

**編集するのは `.in` だけ。** `lock-*.txt` は自動生成物なので手で触らない。

GUI はOS別lockから意図的に分離している。推論結果へ影響する torch / ultralytics を
固定したまま、GUIのセキュリティ更新を独立して適用できる。

```bash
uv pip compile requirements/server.in --python-version 3.10 \
  --python-platform x86_64-unknown-linux-gnu -o requirements/lock-linux.txt

uv pip compile requirements/macos.in --python-version 3.10 \
  --python-platform aarch64-apple-darwin -o requirements/lock-macos.txt
```

## なぜ Linux と macOS でロックを分けるのか

torch のビルドがプラットフォームごとに異なるため。Linux 版は CUDA ランタイム
（`nvidia-*-cu11`）を同梱し、macOS 版は Metal (MPS) を使う。1つのロックには
まとめられない。

バージョンずれは**推論結果を変える**（特に ultralytics）。両ロックは torch 以外を
同じバージョンに揃えてあるので、機械が違ってもゴールデン回帰テストが同じ答えを出す。
