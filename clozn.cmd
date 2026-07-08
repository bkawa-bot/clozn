@echo off
rem clozn launcher (Windows) -- runs the stdlib-only CLI with whatever python is on PATH.
pushd "%~dp0"
python -m clozn %*
popd
