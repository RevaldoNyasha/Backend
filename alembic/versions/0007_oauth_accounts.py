"""oauth accounts (sign in with Google / GitHub)"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0007_oauth_accounts'
down_revision: str | None = '0006_result_details'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('oauth_accounts',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('provider', sa.String(length=20), nullable=False),
    sa.Column('subject', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_oauth_accounts_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_oauth_accounts')),
    sa.UniqueConstraint('provider', 'subject', name=op.f('uq_oauth_accounts_provider'))
    )
    with op.batch_alter_table('oauth_accounts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_oauth_accounts_user_id'), ['user_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('oauth_accounts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_oauth_accounts_user_id'))

    op.drop_table('oauth_accounts')
