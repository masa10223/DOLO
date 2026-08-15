"""計算デバイスの検出と解決。

リファクタ前は ``device="cuda:3"`` がコードに直接書かれており、GPU が4枚
以上刺さった特定のサーバー以外では動かなかった。macOS でも Windows でも
CUDA 無しの Linux でも動くよう、ここで一元的に解決する。

torch は **関数の中で遅延 import する**。``import dolo.device`` 自体は torch
が入っていない環境でも成功し、GUI の起動を遅らせない。
"""

from __future__ import annotations

import warnings

__all__ = [
    "torch_available",
    "available_devices",
    "resolve_device",
    "describe_devices",
]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(
            "この処理には PyTorch が必要です。`pip install 'dolo[torch]'` を実行してください。"
        ) from exc
    return torch


def torch_available() -> bool:
    """torch が import できるか。GUI が推論機能を出し分けるのに使う。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def available_devices() -> list[str]:
    """このマシンで実際に使えるデバイス名の一覧。

    先頭ほど高速な想定で並ぶ。torch が無い場合は ``["cpu"]`` を返す。

    Returns
    -------
    list[str]
        例: ``["cuda:0", "cuda:1", "cpu"]`` / ``["mps", "cpu"]`` / ``["cpu"]``
    """
    if not torch_available():
        return ["cpu"]

    torch = _torch()
    devices: list[str] = []

    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    # Apple Silicon の Metal バックエンド
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")

    devices.append("cpu")
    return devices


def resolve_device(requested: str | int | None = None, strict: bool = False) -> str:
    """要求されたデバイスを、このマシンで実際に使える名前へ解決する。

    Parameters
    ----------
    requested
        ``None`` または ``"auto"`` なら最速のものを自動選択する。
        ``0`` や ``"0"`` は ``"cuda:0"`` と解釈する。
        ``"cuda:3"`` のような明示指定が使えない場合の扱いは ``strict`` 次第。
    strict
        True なら使えないデバイスを要求した時点で ``ValueError`` を投げる。
        False（既定）なら警告を出して利用可能なものへフォールバックする。
        GUI からは strict=False にして、警告文をログに出すとよい。

    Returns
    -------
    str
        ``"cuda:0"`` / ``"mps"`` / ``"cpu"`` のいずれか。

    Examples
    --------
    >>> resolve_device("auto")     # doctest: +SKIP
    'cuda:0'
    >>> resolve_device("cuda:3")   # CUDA の無い Mac で   # doctest: +SKIP
    'mps'
    """
    devices = available_devices()

    # 自動選択
    if requested is None or (isinstance(requested, str) and requested.lower() in ("auto", "")):
        return devices[0]

    # 整数・数字文字列は CUDA のインデックスとみなす（Ultralytics の慣習に合わせる）
    if isinstance(requested, int):
        name = f"cuda:{requested}"
    elif requested.isdigit():
        name = f"cuda:{requested}"
    else:
        name = requested.strip().lower()

    if name == "cuda":
        name = "cuda:0"

    if name in devices:
        return name

    if strict:
        raise ValueError(f"デバイス {requested!r} は使えません。利用可能: {', '.join(devices)}")

    fallback = devices[0]
    warnings.warn(
        f"デバイス {requested!r} は使えないため {fallback!r} を使います。"
        f"（利用可能: {', '.join(devices)}）",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


def describe_devices() -> str:
    """人間向けのデバイス一覧。GUI の設定画面やログの先頭に出す用。"""
    if not torch_available():
        return "PyTorch が未インストールです（CPU のみ、推論・学習は実行できません）"

    torch = _torch()
    lines = [f"PyTorch {torch.__version__}"]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            lines.append(f"  cuda:{i}  {props.name}  {total_gb:.1f} GB")
    else:
        lines.append("  CUDA: 利用不可")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        lines.append("  mps      Apple Silicon GPU")

    lines.append("  cpu")
    return "\n".join(lines)
