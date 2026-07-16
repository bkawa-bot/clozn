# Packaging marker only -- makes `protocol/` (repo root) a REGULAR package when the wheel remaps it to
# `clozn.protocol_fixtures` (see ../setup.py), so a future consumer's `importlib.resources.files(
# "clozn.protocol_fixtures")` gets a real `pathlib.Path` instead of a namespace-package `MultiplexedPath`
# -- see studio/__init__.py's docstring for the concrete bug this avoids. Never imported for behavior;
# contains no code.
