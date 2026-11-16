"""widen_change_summary_to_text

Revision ID: 544f91af7a3f
Revises: 07e2c865c555
Create Date: 2026-11-15 19:31:55.392756

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '544f91af7a3f'
down_revision: Union[str, Sequence[str], None] = '07e2c865c555'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        'video_step_revisions',
        'change_summary',
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        'video_step_revisions',
        'change_summary',
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
