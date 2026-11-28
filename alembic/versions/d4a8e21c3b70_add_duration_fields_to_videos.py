"""add_duration_fields_to_videos

Revision ID: d4a8e21c3b70
Revises: c7a2e81d4b60
Create Date: 2026-11-27 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4a8e21c3b70"
down_revision: Union[str, Sequence[str], None] = "c7a2e81d4b60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add duration_mode and target_duration to videos table."""
    with op.batch_alter_table("videos") as batch_op:
        batch_op.add_column(
            sa.Column("duration_mode", sa.String(length=20), server_default="auto", nullable=False)
        )
        batch_op.add_column(
            sa.Column("target_duration", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema: remove duration fields from videos table."""
    with op.batch_alter_table("videos") as batch_op:
        batch_op.drop_column("target_duration")
        batch_op.drop_column("duration_mode")
