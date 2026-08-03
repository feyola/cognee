"""Add pipeline run status

Revision ID: 1d0bb7fede17
Revises: 482cd6517ce4
Create Date: 2025-05-19 10:58:15.993314
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1d0bb7fede17"
down_revision: Union[str, None] = "482cd6517ce4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "482cd6517ce4"


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        type_exists = bind.execute(
            sa.text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pipelinerunstatus')")
        ).scalar()
        if not type_exists:
            return

        op.execute(
            "ALTER TYPE pipelinerunstatus ADD VALUE IF NOT EXISTS 'DATASET_PROCESSING_INITIATED'"
        )


def downgrade() -> None:
    pass
