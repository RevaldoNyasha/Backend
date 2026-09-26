"""llm usage (Groq quota tracking)"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0008_llm_usage'
down_revision: str | None = '0007_oauth_accounts'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('llm_usage',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('provider', sa.String(length=20), nullable=False),
    sa.Column('model', sa.String(length=120), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('streamed', sa.Boolean(), nullable=False),
    sa.Column('estimated_input_tokens', sa.Integer(), nullable=False),
    sa.Column('max_output_tokens', sa.Integer(), nullable=False),
    sa.Column('input_tokens', sa.Integer(), nullable=True),
    sa.Column('output_tokens', sa.Integer(), nullable=True),
    sa.Column('total_tokens', sa.Integer(), nullable=False),
    sa.Column('error', sa.String(length=300), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_llm_usage'))
    )
    with op.batch_alter_table('llm_usage', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_llm_usage_created_at'), ['created_at'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('llm_usage', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_llm_usage_created_at'))

    op.drop_table('llm_usage')
