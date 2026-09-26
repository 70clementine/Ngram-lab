"""在 Windows/Conda 中启动有资源上限的 Docker；不要求 Windows 安装 kenlm。"""
from pathlib import Path
import subprocess
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ngram-lab-kenlm:1"


def main():
    args = sys.argv[1:] or ["--help"]
    cfg = json.loads((ROOT / "kenlm_config.json").read_text(encoding="utf-8"))
    if args[0] == "build":
        started = time.perf_counter()
        out = ROOT / "results/kenlm"
        out.mkdir(parents=True, exist_ok=True)
        command = ["docker", "build", "-t", IMAGE, "-f", "docker/Dockerfile", "."]
        with (out / "docker_build.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            code = proc.wait()
        (out / "docker_build.json").write_text(json.dumps({"seconds": time.perf_counter()-started, "exit_code": code,
              "note": "首次构建主要受网络影响；缓存构建耗时不能当作首次安装耗时"}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(code)
    command = ["docker", "run", "--rm", "--cpus", str(cfg["container_cpus"]),
               "--memory", cfg["container_memory"], "--memory-swap", cfg["container_memory"],
               "--mount", f"type=bind,source={ROOT},target=/workspace", IMAGE]
    if args[0] == "test":
        command += ["python", "-m", "pytest", "-q"]
    elif args[0] == "audit":
        command += ["python", "scripts/audit_kenlm.py"]
    else:
        command += ["python", "-m", "ngram_lab.kenlm_cli", *args]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
