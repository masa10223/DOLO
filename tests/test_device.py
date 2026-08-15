"""デバイス解決のテスト。torch の有無を問わず動く（無ければ cpu にたたみ込まれる）。"""

from __future__ import annotations

import pytest

from dolo import device as dev


class FakeMPS:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeCUDA:
    def __init__(self, count):
        self._count = count

    def is_available(self):
        return self._count > 0

    def device_count(self):
        return self._count


class FakeTorch:
    __version__ = "2.2.2"

    def __init__(self, cuda_count=0, mps=False):
        self.cuda = FakeCUDA(cuda_count)
        self.backends = type("B", (), {"mps": FakeMPS(mps)})()


@pytest.fixture
def fake_env(monkeypatch):
    """torch の構成を差し替えるヘルパ。"""

    def _apply(cuda_count=0, mps=False, torch_installed=True):
        monkeypatch.setattr(dev, "torch_available", lambda: torch_installed)
        if torch_installed:
            monkeypatch.setattr(dev, "_torch", lambda: FakeTorch(cuda_count, mps))

    return _apply


# --------------------------------------------------------------------------
# available_devices
# --------------------------------------------------------------------------
def test_no_torch_gives_cpu_only(fake_env):
    fake_env(torch_installed=False)
    assert dev.available_devices() == ["cpu"]


def test_multi_gpu_linux_lists_all_cuda_devices(fake_env):
    fake_env(cuda_count=4)
    assert dev.available_devices() == ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"]


def test_apple_silicon_lists_mps(fake_env):
    fake_env(cuda_count=0, mps=True)
    assert dev.available_devices() == ["mps", "cpu"]


def test_cpu_only_machine(fake_env):
    fake_env(cuda_count=0, mps=False)
    assert dev.available_devices() == ["cpu"]


def test_cpu_is_always_last_resort(fake_env):
    fake_env(cuda_count=2, mps=True)
    assert dev.available_devices()[-1] == "cpu"


# --------------------------------------------------------------------------
# resolve_device — 自動選択
# --------------------------------------------------------------------------
@pytest.mark.parametrize("requested", [None, "auto", "AUTO", ""])
def test_auto_picks_fastest_available(fake_env, requested):
    fake_env(cuda_count=2)
    assert dev.resolve_device(requested) == "cuda:0"


def test_auto_prefers_mps_over_cpu_on_mac(fake_env):
    fake_env(cuda_count=0, mps=True)
    assert dev.resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu(fake_env):
    fake_env(cuda_count=0, mps=False)
    assert dev.resolve_device("auto") == "cpu"


# --------------------------------------------------------------------------
# resolve_device — 明示指定
# --------------------------------------------------------------------------
def test_explicit_cuda_device_is_honoured(fake_env):
    fake_env(cuda_count=4)
    assert dev.resolve_device("cuda:3") == "cuda:3"


@pytest.mark.parametrize("requested,expected", [(0, "cuda:0"), (2, "cuda:2"), ("1", "cuda:1")])
def test_integer_means_cuda_index(fake_env, requested, expected):
    fake_env(cuda_count=4)
    assert dev.resolve_device(requested) == expected


def test_bare_cuda_means_first_gpu(fake_env):
    fake_env(cuda_count=2)
    assert dev.resolve_device("cuda") == "cuda:0"


def test_device_name_is_case_insensitive(fake_env):
    fake_env(cuda_count=1)
    assert dev.resolve_device("CUDA:0") == "cuda:0"


# --------------------------------------------------------------------------
# resolve_device — フォールバック（ここが cuda:3 ハードコードの救済）
# --------------------------------------------------------------------------
def test_hardcoded_cuda3_falls_back_on_mac_with_warning(fake_env):
    """既存コードの device='cuda:3' が Mac でも動くようになる。"""
    fake_env(cuda_count=0, mps=True)
    with pytest.warns(RuntimeWarning, match="cuda:3"):
        assert dev.resolve_device("cuda:3") == "mps"


def test_out_of_range_gpu_falls_back_to_first_gpu(fake_env):
    """GPU 2枚のマシンで cuda:3 を要求した場合。"""
    fake_env(cuda_count=2)
    with pytest.warns(RuntimeWarning):
        assert dev.resolve_device("cuda:3") == "cuda:0"


def test_strict_mode_raises_instead_of_falling_back(fake_env):
    fake_env(cuda_count=0, mps=True)
    with pytest.raises(ValueError, match="利用可能"):
        dev.resolve_device("cuda:3", strict=True)


def test_strict_error_lists_available_devices(fake_env):
    fake_env(cuda_count=2)
    with pytest.raises(ValueError) as excinfo:
        dev.resolve_device("cuda:9", strict=True)
    assert "cuda:0" in str(excinfo.value)
    assert "cuda:1" in str(excinfo.value)


def test_unknown_device_name_falls_back(fake_env):
    fake_env(cuda_count=1)
    with pytest.warns(RuntimeWarning):
        assert dev.resolve_device("tpu") == "cuda:0"


# --------------------------------------------------------------------------
# describe_devices
# --------------------------------------------------------------------------
def test_describe_without_torch_is_explicit(fake_env):
    fake_env(torch_installed=False)
    assert "PyTorch" in dev.describe_devices()


def test_describe_mentions_mps_on_apple_silicon(fake_env):
    fake_env(cuda_count=0, mps=True)
    assert "mps" in dev.describe_devices()


# --------------------------------------------------------------------------
# 実環境（スタブ無し）— 何であれ壊れないこと
# --------------------------------------------------------------------------
def test_real_environment_resolves_without_error():
    resolved = dev.resolve_device("auto")
    assert resolved in dev.available_devices()


def test_real_environment_always_offers_cpu():
    assert "cpu" in dev.available_devices()
