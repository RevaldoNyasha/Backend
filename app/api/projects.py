from collections.abc import Sequence
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.dependencies import CurrentProject, CurrentUser, DbSession
from app.models.document import KIND_FOCUS_GROUP, KIND_INTERVIEW, Document
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectResponse
from app.services.project import (
    create_project,
    delete_project,
    get_project_by_name,
    list_projects,
)

router = APIRouter(tags=["projects"])


@dataclass
class ProjectCounts:
    document_count: int = 0
    interview_count: int = 0
    focus_group_count: int = 0
    kinds: dict[str, int] = field(default_factory=dict)


def collect_document_counts(db: Session, project_ids: Sequence[str]) -> dict[str, ProjectCounts]:
    if not project_ids:
        return {}

    rows = db.execute(
        select(Document.project_id, Document.kind, func.count(Document.id))
        .where(Document.project_id.in_(project_ids))
        .group_by(Document.project_id, Document.kind)
    )

    counts: dict[str, ProjectCounts] = {}
    for project_id, kind, total in rows:
        entry = counts.setdefault(project_id, ProjectCounts())
        entry.document_count += total
        entry.kinds[kind] = total
        if kind == KIND_INTERVIEW:
            entry.interview_count += total
        elif kind == KIND_FOCUS_GROUP:
            entry.focus_group_count += total

    return counts


def to_project_response(project: Project, counts: ProjectCounts | None = None) -> ProjectResponse:
    totals = counts or ProjectCounts()
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        document_count=totals.document_count,
        participant_count=0,
        interview_count=totals.interview_count,
        focus_group_count=totals.focus_group_count,
        themes=[],
        codes=[],
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.get("/projects", response_model=list[ProjectResponse])
def index(db: DbSession, current_user: CurrentUser) -> list[ProjectResponse]:
    projects = list_projects(db, user_id=current_user.id)
    counts = collect_document_counts(db, [project.id for project in projects])

    return [to_project_response(project, counts.get(project.id)) for project in projects]


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create(payload: ProjectCreate, db: DbSession, current_user: CurrentUser) -> ProjectResponse:
    name = payload.name.strip()
    if get_project_by_name(db, current_user.id, name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already in use",
        )

    try:
        project = create_project(
            db,
            user_id=current_user.id,
            name=name,
            description=payload.description.strip(),
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already in use",
        ) from None

    return to_project_response(project)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def show(project_id: str, current_project: CurrentProject, db: DbSession) -> ProjectResponse:
    counts = collect_document_counts(db, [current_project.id])
    return to_project_response(current_project, counts.get(current_project.id))


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def destroy(project_id: str, current_project: CurrentProject, db: DbSession) -> None:
    delete_project(db, current_project)
