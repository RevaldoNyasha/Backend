from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.project import Project
from app.services.storage import get_storage


def create_project(db: Session, user_id: str, name: str, description: str) -> Project:
    project = Project(user_id=user_id, name=name, description=description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: str, user_id: str) -> Project | None:
    return db.scalar(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )


def get_project_by_name(db: Session, user_id: str, name: str) -> Project | None:
    return db.scalar(select(Project).where(Project.user_id == user_id, Project.name == name))


def list_projects(db: Session, user_id: str) -> list[Project]:
    return list(
        db.scalars(
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.updated_at.desc(), Project.created_at.desc())
        )
    )


def mark_project_updated(project: Project) -> None:
    project.updated_at = datetime.now(UTC)


def delete_project(db: Session, project: Project) -> None:
    # SQLite does not enforce ON DELETE CASCADE by default, so remove the
    # project's documents explicitly, then its folder of stored files.
    project_id = project.id
    db.execute(delete(Document).where(Document.project_id == project_id))
    db.delete(project)
    db.commit()
    get_storage().delete_prefix(project_id)
