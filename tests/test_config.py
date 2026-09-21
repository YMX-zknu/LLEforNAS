from pathlib import Path

from lnas.config import load_config


def test_load_config_with_override():
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    config = load_config(path, ["dataset.timesteps=3", "proxy.stable_only=false"])
    assert config.dataset.timesteps == 3
    assert config.proxy.stable_only is False


def test_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: 1\n", encoding="utf-8")
    try:
        load_config(path)
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("Unknown configuration keys must fail")
