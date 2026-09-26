from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentDocument, CurrentProject, DbSession
from app.models.result import RESULT_STATUS_READY, Result
from app.schemas.result import SummaryResponse
from app.services.processing_worker import notify_generation_worker
from app.services.summarizer import (
    SummaryAlreadyGenerating,
    SummaryNotAvailable,
    get_summary,
    request_summary,
)

router = APIRouter(tags=["summaries"])

SUMMARY_PATH = "/projects/{project_id}/documents/{document_id}/summary"


def to_response(summary: Result) -> SummaryResponse:
    return SummaryResponse(
        document_id=summary.document_id or "",
        status=summary.status,
        content=summary.content if summary.status == RESULT_STATUS_READY else None,
        error=summary.error,
        model=summary.model_name,
        created_at=summary.created_at,
        completed_at=summary.completed_at,
    )


@router.post(SUMMARY_PATH, response_model=SummaryResponse, status_code=status.HTTP_202_ACCEPTED)
def create_summary(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> SummaryResponse:
    """Start (or redo) the document's summary; poll `GET` for the result."""
    try:
        summary = request_summary(db, current_project, current_document)
    except SummaryNotAvailable as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from None
    except SummaryAlreadyGenerating:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A summary is already being generated for this document",
        ) from None
    notify_generation_worker()
    return to_response(summary)


@router.get(SUMMARY_PATH, response_model=SummaryResponse)
def read_summary(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> SummaryResponse:
    summary = get_summary(db, current_document)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No summary has been generated yet"
        )
    return to_response(summary)
