# 0002 Use uv, Ruff, and mypy

Accepted on 2026-11-26.

Development starts on Windows and the hosted checks run on Linux. Later milestones share
scientific and web dependencies and must run on CPU.

Use uv with a shared lockfile and Python 3.12 by default. Resolve torch and torchvision
from the explicit CPU index for Linux and Windows. Check both platforms during lock
resolution so a dependency update cannot silently remove a required wheel. Use Ruff for
linting and formatting, strict mypy for the package, and pytest for offline checks.

Dependency updates require a lockfile change. The four CI jobs report lint, type checking,
unit tests, and smoke tests independently. Linux users who download the full crack dataset
also need a RAR-capable extraction tool.
