"""Clean-room install proof: build clozn's wheel, install it into a brand-new venv, then prove the
console entry point, `python -m clozn`, and `import clozn` all work OUTSIDE this repo checkout -- no
CLOZN_* env vars pointing back at it, no repo-relative sys.path hacks, cwd is a scratch temp dir. This is
the actual bar docs/BACKLOG.md Sec.2 sets ("pip install into a clean env yields a working clozn with no
repo-path hacks"); a passing run here is what "packaging works" means for this project, not just "the
wheel built".

    python scripts/release/clean_room_install_test.py
    python scripts/release/clean_room_install_test.py --keep      # leave the venv+dist dirs for inspection

Deliberately stdlib-only (venv, subprocess, tempfile) -- this has to run before anything else is proven to
work, so it can't lean on the package it's testing. Requires the `build` package be importable by THIS
interpreter (the wheel is built with `sys.executable`, not the target venv) -- CI installs it as a step;
locally: `pip install build`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _venv_python(venv_dir: str) -> str:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv_dir, bin_dir, exe)


def _venv_console_script(venv_dir: str, name: str) -> str:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = f"{name}.exe" if os.name == "nt" else name
    return os.path.join(venv_dir, bin_dir, exe)


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="don't delete the scratch venv/dist dirs afterward")
    args = ap.parse_args(argv)

    scratch = tempfile.mkdtemp(prefix="clozn-clean-room-")
    dist_dir = os.path.join(scratch, "dist")
    venv_dir = os.path.join(scratch, "venv")
    outside_cwd = os.path.join(scratch, "elsewhere")   # proves nothing depends on REPO being cwd
    os.makedirs(outside_cwd, exist_ok=True)
    print(f"scratch dir: {scratch}")

    try:
        # 1) Build the wheel with THIS interpreter's build tooling -- the target venv only ever needs pip.
        build = _run([sys.executable, "-m", "build", "--wheel", "--outdir", dist_dir], cwd=REPO)
        if build.returncode != 0:
            print("FAIL: wheel build failed", file=sys.stderr)
            return 1
        wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
        if not wheels:
            print("FAIL: no wheel produced", file=sys.stderr)
            return 1
        wheel_path = os.path.join(dist_dir, wheels[0])
        print(f"built {wheel_path}")

        # 2) A brand-new venv -- not the interpreter running this script, so nothing it already has
        #    installed (or an editable-install path hack) can paper over a packaging bug.
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
        vpy = _venv_python(venv_dir)

        # 3) Install ONLY the wheel -- no `-e`, no repo on PYTHONPATH, no CLOZN_* overrides.
        install = _run([vpy, "-m", "pip", "install", "--no-input", wheel_path])
        if install.returncode != 0:
            print("FAIL: pip install of the wheel failed", file=sys.stderr)
            return 1

        checks = []

        def check(name, cmd):
            result = _run(cmd, cwd=outside_cwd)
            ok = result.returncode == 0
            checks.append((name, ok))
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
            return result

        # `import clozn` must work with NO repo on sys.path and NOT running from inside the repo.
        check('python -c "import clozn"', [vpy, "-c", "import clozn; print('clozn at', clozn.__file__)"])

        # `clozn version` / `clozn doctor` via `python -m clozn` (the documented entry point clozn.cmd /
        # clozn.sh both use) -- doctor must not crash even though nothing is installed/running yet.
        check("python -m clozn version", [vpy, "-m", "clozn", "version"])
        check("python -m clozn doctor", [vpy, "-m", "clozn", "doctor"])

        # The actual `clozn` console script pip put on the venv's PATH (not `python -m`) -- this is the
        # literal `[project.scripts]` entry point pyproject.toml declares.
        console = _venv_console_script(venv_dir, "clozn")
        if os.path.isfile(console):
            check("console script `clozn version`", [console, "version"])
            check("console script `clozn doctor`", [console, "doctor"])
        else:
            print(f"FAIL: console entry point not found at {console}", file=sys.stderr)
            checks.append(("console script exists", False))

        failed = [name for name, ok in checks if not ok]
        print()
        if failed:
            print(f"CLEAN-ROOM INSTALL TEST: FAILED ({len(failed)}/{len(checks)}): {', '.join(failed)}")
            return 1
        print(f"CLEAN-ROOM INSTALL TEST: PASSED ({len(checks)}/{len(checks)})")
        return 0
    finally:
        if args.keep:
            print(f"--keep: leaving {scratch} in place")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
