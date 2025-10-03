"""Entry point for ``python -m rinker``."""
from .cli import main

if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())
