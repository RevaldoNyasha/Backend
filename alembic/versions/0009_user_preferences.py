"""users.preferences"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0009_user_preferences'
down_revision: str | None = '0008_llm_usage'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('preferences', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('preferences')
