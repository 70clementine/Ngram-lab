"""按 Git 忽略规则打包可分享的项目，不附带语料全文和 pickle 模型。"""
import argparse
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    files = sorted(set(p.decode("utf-8") for p in raw.split(b"\0") if p))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = root / name
            if path.is_file() and path.resolve() != args.output.resolve():
                archive.write(path, "ngram/" + name)
    print(f"Packaged {len(files)} files -> {args.output}")


if __name__ == "__main__":
    main()
