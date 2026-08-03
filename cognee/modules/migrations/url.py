"""Safe database URL rendering for migration diagnostics."""

from sqlalchemy.engine import URL


def render_database_url_for_logging(url: URL) -> str:
    """Render a useful connection target without exposing its password."""

    return url.render_as_string(hide_password=True)
