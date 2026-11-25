"""Chạy toàn bộ bộ kiểm thử tự động trong scripts/test_*.py bằng một lệnh duy nhất.

Cách dùng:
    python scripts/run_all_tests.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tests = sorted((ROOT / "scripts").glob("test_*.py"))
    if not tests:
        raise SystemExit("Không tìm thấy file test nào trong scripts/test_*.py")

    failed = []
    t0_all = time.perf_counter()
    print(f"Chạy {len(tests)} bộ kiểm thử trong scripts/...\n")
    for path in tests:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), capture_output=True, text=True)
        dt = time.perf_counter() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"[{status}] {path.name:<42s} ({dt:5.2f}s)")
        if proc.returncode != 0:
            failed.append(path.name)
            if proc.stdout:
                print(proc.stdout.rstrip())
            if proc.stderr:
                print(proc.stderr.rstrip())

    total_dt = time.perf_counter() - t0_all
    print(f"\nTổng kết: {len(tests) - len(failed)}/{len(tests)} PASS trong {total_dt:.2f}s.")
    if failed:
        raise SystemExit(f"Các test lỗi: {', '.join(failed)}")


if __name__ == "__main__":
    main()
