"""
Local Jarvis - Analytics Package
"""

from src.analytics.logger import (
    DEFAULT_DB_PATH,
    flush_logging,
    init_db,
    log_interaction,
    shutdown_logging,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "init_db",
    "log_interaction",
    "flush_logging",
    "shutdown_logging",
    "export_to_csv",
]


def __getattr__(name: str):
    if name in ("export_to_csv", "DEFAULT_CSV_PATH"):
        import src.analytics.export_csv as exp
        return getattr(exp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
