from pathlib import Path

import pytest

from vnedu_cot.providers import load_model_configs
from vnedu_cot.schemas.core import ApiGateway, ModelAccessTier


def test_load_model_configs_from_configs_models_dir() -> None:
    configs = load_model_configs(Path("configs/models"))
    model_ids = [c.model_id for c in configs]
    assert "deepseek/deepseek-r1-0528" in model_ids
    assert len(configs) == 1


def test_load_model_configs_single_file() -> None:
    configs = load_model_configs(Path("configs/models/deepseek_r1.toml"))
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.model_id == "deepseek/deepseek-r1-0528"
    assert cfg.provider == "DeepInfra"
    assert cfg.allow_fallbacks is False
    assert cfg.access_tier == ModelAccessTier.FULL_CONTROL
    assert cfg.api_gateway == ApiGateway.OPENROUTER


def test_load_model_configs_rejects_non_toml(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "models.jsonl"
    jsonl_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a .toml model config file"):
        load_model_configs(jsonl_file)

