# DOLO の GPU サーバー環境を再現する Docker イメージ。
#
#   docker build -t dolo:dev .
#   docker run --gpus all -it --rm -v "$PWD":/workspace dolo:dev bash
#
# CUDA 11.7 系は torch 2.0.0（requirements/lock-linux.txt が固定している版）に
# 合わせたもの。torch を上げる際はこのベースイメージも一緒に上げること。

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ultralytics が対話的な設定やテレメトリを出さないようにする
    YOLO_AUTOINSTALL=false \
    MPLBACKEND=Agg

# opencv と動画I/Oに必要な共有ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv（pip より大幅に高速）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.10 $VIRTUAL_ENV

WORKDIR /workspace

# 依存だけ先に入れてレイヤーキャッシュを効かせる
COPY requirements/ /workspace/requirements/
RUN uv pip install -r requirements/lock-linux.txt
RUN uv pip install -r requirements/gui.in

# パッケージ本体
COPY pyproject.toml README.md /workspace/
COPY dolo/ /workspace/dolo/
RUN uv pip install --no-deps -e .

COPY . /workspace/

# 学習データや動画はイメージに含めず、実行時に mount する
VOLUME ["/data"]
ENV DOLO_DATA_DIR=/data
EXPOSE 8080

CMD ["dolo", "gui", "--host", "0.0.0.0", "--port", "8080", "--no-open"]
