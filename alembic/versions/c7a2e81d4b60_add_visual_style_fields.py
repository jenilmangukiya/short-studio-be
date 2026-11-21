"""add_visual_style_fields

Revision ID: c7a2e81d4b60
Revises: 544f91af7a3f
Create Date: 2026-11-21 14:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c7a2e81d4b60"
down_revision: Union[str, Sequence[str], None] = "544f91af7a3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add visual style fields to channels and videos."""
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    with op.batch_alter_table("channels") as batch_op:
        batch_op.add_column(sa.Column("default_visual_style_id", sa.String(length=64), nullable=True))
    with op.batch_alter_table("videos") as batch_op:
        batch_op.add_column(sa.Column("visual_style_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("visual_style_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("visual_style_snapshot", json_type, nullable=True))


def downgrade() -> None:
    """Downgrade schema: remove visual style fields."""
    with op.batch_alter_table("videos") as batch_op:
        batch_op.drop_column("visual_style_snapshot")
        batch_op.drop_column("visual_style_version")
        batch_op.drop_column("visual_style_id")
    with op.batch_alter_table("channels") as batch_op:
        batch_op.drop_column("default_visual_style_id")
