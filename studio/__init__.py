# Packaging marker only -- makes `studio/` (repo root) a REGULAR package when the wheel remaps it to
# `clozn.studio` (see ../setup.py). Without this, `clozn.studio` installs as an implicit PEP 420
# namespace package, and `importlib.resources.files("clozn.studio")` then returns a `MultiplexedPath`
# rather than a real `pathlib.Path` -- `str()` of one is not a usable filesystem path, which silently
# broke `clozn/server/config.py`'s packaged-mode Studio asset lookup (verified empirically via
# scripts/release/clean_room_install_test.py: "heavn/index.html is missing" even though the file was
# genuinely inside the installed wheel). Never imported for behavior; contains no code.
