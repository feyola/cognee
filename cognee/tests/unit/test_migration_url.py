from sqlalchemy.engine import make_url

from cognee.modules.migrations.url import render_database_url_for_logging


def test_migration_database_url_redacts_password():
    rendered = render_database_url_for_logging(
        make_url("postgresql+asyncpg://cognee:do-not-log-me@postgres:5432/cognee")
    )

    assert "do-not-log-me" not in rendered
    assert rendered == "postgresql+asyncpg://cognee:***@postgres:5432/cognee"
