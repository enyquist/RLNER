# standard libaries
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"


def main() -> None:
    """Prepare data from raw format"""
    pass


if __name__ == "__main__":
    main()
