"""transcription: chunk timestamps and document progress"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0010_transcription'
down_revision: str | None = '0009_user_preferences'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('processing_progress', sa.String(length=120), nullable=True))
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('start_seconds', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('end_seconds', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('end_seconds')
        batch_op.drop_column('start_seconds')
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('processing_progress')
