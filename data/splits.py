from pathlib import Path

import yaml

SPLIT_FILE = Path(__file__).resolve().parents[1] / "configs/split.yaml"


def load_hidden_pairs(path: Path = SPLIT_FILE) -> list[str]:
    """The frozen hidden test pairs."""
    return yaml.safe_load(path.read_text())["hidden_pairs"]