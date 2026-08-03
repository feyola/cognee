from importlib import import_module
from types import SimpleNamespace


migration = import_module("cognee.alembic.versions.1d0bb7fede17_add_pipeline_run_status")


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeBind:
    def __init__(self, dialect_name, type_exists=None):
        self.dialect = SimpleNamespace(name=dialect_name)
        self.type_exists = type_exists
        self.queries = []

    def execute(self, query):
        self.queries.append(str(query))
        return ScalarResult(self.type_exists)


def test_pipeline_status_migration_skips_missing_postgres_enum(monkeypatch):
    bind = FakeBind("postgresql", type_exists=False)
    statements = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    assert len(bind.queries) == 1
    assert statements == []


def test_pipeline_status_migration_updates_existing_postgres_enum(monkeypatch):
    bind = FakeBind("postgresql", type_exists=True)
    statements = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    assert statements == [
        "ALTER TYPE pipelinerunstatus ADD VALUE IF NOT EXISTS 'DATASET_PROCESSING_INITIATED'"
    ]


def test_pipeline_status_migration_is_noop_outside_postgres(monkeypatch):
    bind = FakeBind("sqlite")
    statements = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    assert bind.queries == []
    assert statements == []
