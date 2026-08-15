[English](README.md) | **日本語**

# DOLO

**D**rosophila tracking with **YOLO** **P**ose — *Drosophila* 動画から姿勢推定・個体追跡・行動解析を行うソフトウェアです。

ブラウザ上の GUI で動画を選び、学習済みモデルで追跡し、軌跡 CSV / アノテーション付き動画などを出力できます。  
この README は、**ターミナルや Git に慣れていない方**でも一通り動かせるように書いています。

関連論文（投稿中）:

> **Chemosensory input suppresses cannibalism by stabilizing social feeding boundaries in Drosophila larvae**  
> Nagisa Matsuda-Watanabe, Masato Tsutsumi, Misako Okumura, and Takahiro Chihara

---

## いまからやること（全体像）

1. パソコンに **Git** と **uv**（Python の入れやすい道具）を入れる  
2. GitHub から DOLO のプログラムを **`git clone`** する  
3. Zenodo から **`best.pt`（重み）** と **`test_movie.mov`（試し用動画）** をダウンロードする  
4. ダウンロードしたファイルを、clone したフォルダの決まった場所に置く  
5. 依存パッケージを入れて、**`dolo gui`** でブラウザを開く  

所要時間の目安: 初回 15–40 分（ネット速度と PC 性能による）。  
`best.pt` は約 **120 MB** あります。

対応 OS: **macOS** または **Linux**（Windows は未整備。WSL2 利用は自己責任で）。

---

## 0. 用語の超短い説明

| 言葉 | 意味 |
|------|------|
| **ターミナル** | 黒い画面にコマンドを打ち込むアプリ（macOS なら「ターミナル」） |
| **リポジトリ** | GitHub 上のプロジェクト一式。clone すると自分の PC にコピーされる |
| **clone** | インターネット上のプログラムを、自分の PC にダウンロードすること |
| **重み (`best.pt`)** | 学習済みモデルのファイル。これがないと推論（追跡）できない |
| **仮想環境 (`.venv`)** | このプロジェクト専用の Python 置き場。他の実験と混ざらない |

コマンドは **1行ずつ**コピーして Enter で実行してください。エラーが出たら、その全文を控えて相談すると早いです。

---

## 1. Git を入れる（まだの人だけ）

### macOS

ターミナルを開き、次を実行します。

```bash
git --version
```

`git version ...` と出れば OK です。  
「Command Line Developer Tools を入れますか？」と聞かれたら **Install** を選び、終わってからもう一度同じコマンドを実行してください。

Homebrew を使う場合の例:

```bash
brew install git
```

### Linux（Ubuntu の例）

```bash
sudo apt update
sudo apt install -y git
git --version
```

---

## 2. DOLO を `git clone` する

好きな作業フォルダに移動してから、次を実行します（例はホーム直下）。

```bash
cd ~
git clone https://github.com/masa10223/DOLO.git
cd DOLO
```

成功すると、`DOLO` というフォルダができ、その中にプログラム一式があります。  
以降のコマンドは、**必ずこの `DOLO` フォルダの中**で実行してください。

今どこにいるか確認するコマンド:

```bash
pwd
ls
```

`pwd` の末尾が `.../DOLO` で、`ls` に `README.md` や `dolo` が見えれば正しい場所です。

> **更新したいとき（2回目以降）**  
> 同じフォルダで `git pull` とすると、GitHub 上の新しい変更を取り込めます。

---

## 3. Zenodo から重みと試し用動画をダウンロードする

モデルの重みは GitHub には入れていません（ファイルが大きいため）。  
次の Zenodo レコードからダウンロードしてください。

- DOI: **[10.5281/zenodo.21951363](https://doi.org/10.5281/zenodo.21951363)**  
- ページ: https://doi.org/10.5281/zenodo.21951363  
- 入っているファイル:
  - **`best.pt`** … 標準の学習済み重み（必須）
  - **`test_movie.mov`** … 動作確認用の短い動画（推奨）

### ブラウザでの手順（推奨）

1. 上の DOI リンクをブラウザで開く  
2. ページ内の **Files**（ファイル一覧）を探す  
3. `best.pt` の横のダウンロードボタンを押す（約 120 MB）  
4. 同じく `test_movie.mov` もダウンロードする（約 1.2 MB）  
5. ダウンロード先は、多くの場合「ダウンロード」フォルダです

### ファイルの置き場所（ここが一番大事）

Finder（またはファイルマネージャ）で、clone した `DOLO` フォルダを開きます。

| ダウンロードしたファイル | 置く場所 | 完成イメージ |
|--------------------------|----------|----------------|
| `best.pt` | **`DOLO` フォルダの直下**（`README.md` と同じ階層） | `DOLO/best.pt` |
| `test_movie.mov` | 同じ直下で OK（好きな場所でも可） | `DOLO/test_movie.mov` |

ターミナルで確認する例（macOS、ダウンロードフォルダから移す場合）:

```bash
cd ~/DOLO
mv ~/Downloads/best.pt .
mv ~/Downloads/test_movie.mov .
ls -lh best.pt test_movie.mov
```

`best.pt` がだいたい **120M** 前後と表示されれば成功です。

> **置かないとどうなる？**  
> GUI は起動できても、「モデルが必要です」と出たり、推論を開始できません。  
> `dolo doctor` でモデルが見つかるか確認できます。

別の置き方（上級者向け）:

- `models/default.pt` として置く  
- または環境変数 `DOLO_MODEL_PATH=/絶対パス/best.pt` を設定する  

---

## 4. uv（Python 環境用ツール）を入れる

DOLO では [uv](https://docs.astral.sh/uv/) を使って Python とライブラリを入れます。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

表示に従ってターミナルを開き直すか、指示された `source ...` を実行したあと:

```bash
uv --version
```

バージョンが出れば OK です。

macOS で Homebrew を使う場合:

```bash
brew install uv
```

Windows など他環境は [uv の公式インストール手順](https://docs.astral.sh/uv/getting-started/installation/) を参照してください。

---

## 5. DOLO をインストールする

`DOLO` フォルダにいることを確認してから:

```bash
cd ~/DOLO
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
```

プロンプトの先頭に `(.venv)` が付けば、仮想環境に入れています。

次に、OS に合わせて **どちらか一方** を実行します。

**macOS（Apple Silicon / Intel）:**

```bash
uv pip install -r requirements/lock-macos.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

**NVIDIA GPU 付き Linux サーバー:**

```bash
uv pip install -r requirements/lock-linux.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

初回は数分かかることがあります。

### 動作チェック

```bash
dolo doctor
```

`READY` と出れば準備完了です。  
`SETUP REQUIRED` の場合は、表示された不足項目（特に `best.pt` の有無）を直してください。

> ターミナルを閉じて開き直したあと、もう一度 GUI を使うときは、毎回まず次を実行します。  
> `cd ~/DOLO` → `source .venv/bin/activate`

---

## 6. GUI を起動して試す

```bash
cd ~/DOLO
source .venv/bin/activate
dolo gui
```

ブラウザが開かない場合は、自分で次を開きます:  
http://127.0.0.1:8080

1. 画面の案内に従い、動画を選ぶ（試しなら `test_movie.mov`）  
2. 既定モデルに `best.pt` が見えているか確認する  
3. **推論を開始** を押す  

結果は `.dolo/runs/` の下に保存されます（軌跡 CSV、任意で動画 / GIF、設定の控えなど）。

### うまくいかないとき

**ポート 8080 が使われている**

```text
address already in use
```

別ポートで起動します。

```bash
dolo gui --port 8090
```

ブラウザでは http://127.0.0.1:8090 を開きます。

**モデルが見つからない**

- `DOLO/best.pt` があるか、もう一度確認する  
- または明示指定: `dolo gui --model /フルパス/best.pt`

**止めるとき**

GUI を起動したターミナルで `Ctrl+C` を押します。

より詳しい GUI / サーバー運用は [docs/GUI.md](docs/GUI.md)、環境の細部は [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) を参照してください。

> GUI は標準では自分の PC の中（`127.0.0.1`）だけに開きます。インターネットにむき出しで公開しないでください。

---

## できること（概要）

- **ブラウザ GUI** … 動画アップロード、追跡、進捗確認、結果ダウンロード  
- **手動アノテーション連携** … head / mid / tail の CSV ラベル  
- **学習パイプライン** … 動画ごとの fine-tune、複数動画の foundation 学習（上級）  
- **追跡結果の解析** … 接触・相互作用などの下流解析スクリプト  

学習や大規模バッチは GPU サーバー向けです。まずは GUI + `test_movie.mov` で流れを掴むのがおすすめです。

---

## よくある質問

**Q. `best.pt` を GitHub に上げればよくない？**  
A. GitHub は大きなファイル（特に 100 MB 超）を通常の方法では扱えません。そのため Zenodo に置いています。

**Q. clone したのに `best.pt` が無い**  
A. 仕様です。Zenodo から別途ダウンロードして、`DOLO` 直下に置いてください。

**Q. 自分の実験動画で使いたい**  
A. GUI で自分の動画を選べます。撮影条件が大きく違う場合は、追加の学習（fine-tune）が必要になることがあります。

**Q. 開発者が変更を送るとき、どのブランチ？**  
A. 作業中は **`dev`** 向けに PR。安定版だけ **`main`**。迷ったら `dev` です。詳細は下の「開発者向け」へ。

---

## 開発者向け（短く）

```
feature/*  →  PR →  dev  →  (安定したら)  main
```

大きな成果物（動画・アノテーション・`*.pt`）は `.gitignore` 対象です。コミットしないでください。

レガシーな学習・推論スクリプトは `scripts/` 以下にあります。パッケージ化された API / GUI は `dolo/` です。

---

## Citation

公表時には関連原稿を引用し、[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) の利用も明記してください。DOI / 誌面フォーマットは公開後に追記します。

重み・デモ動画のアーカイブ:

> Tsutsumi, M. (2026). Test movie and pre-trained model weight for DOLO. Zenodo.  
> https://doi.org/10.5281/zenodo.21951363

---

## Copyright and license

Copyright © 2024–2026 **Masato Tsutsumi**.  
ソースコードとオリジナル文書は **AGPL-3.0-or-later**（[LICENSE](LICENSE)）です。  
第三者ソフト・モデル重み・データセット・論文・商標は、それぞれの権利に従います。

ソフトウェア設計・実装: **Masato Tsutsumi** ([mtsutsumi@nagoya-u.jp](mailto:mtsutsumi@nagoya-u.jp))  
近況: [研究ページ](https://masa10223.github.io/en/)
