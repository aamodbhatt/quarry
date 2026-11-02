"""Exception hierarchy. Every error a user can see says what failed and how to fix it."""

from __future__ import annotations


class QuarryError(Exception):
    """Base class for every error quarry raises on purpose."""


class ConfigError(QuarryError):
    """The configuration file or the environment is invalid or incomplete."""


class DatabaseError(QuarryError):
    """The database could not be opened or introspected."""


class QueryRejected(QuarryError):
    """The SQL guard refused a statement before it reached the database."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class QueryFailed(QuarryError):
    """The database raised an error while running an allowed statement."""


class QueryTimeout(QueryFailed):
    """The statement ran past its time budget and was interrupted."""


class DatasetError(QuarryError):
    """A benchmark or demo dataset is missing, corrupt or failed its checksum."""


class BenchmarkInterrupted(QuarryError):
    """The model API kept failing; progress is saved and the same command resumes the run."""
