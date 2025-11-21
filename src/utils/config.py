# src/utils/config.py
from pathlib import Path

# Diretório raiz do projeto (onde setup.py está)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"

# Criar pastas automaticamente
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)
