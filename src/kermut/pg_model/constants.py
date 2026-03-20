from importlib import resources
from pathlib import Path

HYDRA_CONFIG_PATH = resources.files("kermut").joinpath("kermut/hydra_configs")
HYDRA_TEMP_CONFIG_PATH = resources.files("kermut").joinpath("pg_model/hydra_configs")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
