"""文件读写和配置；所有路径相对于项目根目录，不依赖当前终端目录。"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def config():
    return read_json(ROOT / "config.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentences(split):
    path = ROOT / "data" / "processed" / f"{split}.txt"
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
