from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentProject, DbSession
from app.models.result import RESULT_STATUS_PROCESSING, RESULT_STATUS_READY, Result
from app.schemas.result import AnalysisDocument, AnalysisResponse
from app.services.analyzer import (
    AnalysisAlreadyRunning,
    AnalysisNotAvailable,
    get_analysis,
    request_analysis,
)
from app.services.processing_worker import notify_generation_worker

router = APIRouter(tags=["analysis"])


def to_response(analysis: Result) -> AnalysisResponse:
    details = analysis.details or {}
    return AnalysisResponse(
        project_id=analysis.project_id,
        status=analysis.status,
        content=analysis.content if analysis.status == RESULT_STATUS_READY else None,
        error=analysis.error,
        model=analysis.model_name,
        progress=details.get("progress") if analysis.status == RESULT_STATUS_PROCESSING else None,
        documents=[AnalysisDocument(**document) for document in details.get("documents", [])],
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
    )


@router.post(
    "/projects/{project_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze(project_id: str, current_project: CurrentProject, db: DbSession) -> AnalysisResponse:
    """Start (or redo) the analysis of all processed documents; poll `GET …/analysis`."""
    try:
        analysis = request_analysis(db, current_project)
    except AnalysisNotAvailable as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from None
    except AnalysisAlreadyRunning:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An analysis is already running for this project",
        ) from None
    notify_generation_worker()
    return to_response(analysis)


@router.get("/projects/{project_id}/analysis", response_model=AnalysisResponse)
def read_analysis(project_id: str, current_project: CurrentProject, db: DbSession):
    analysis = get_analysis(db, current_project)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This project has not been analysed yet"
        )
    return to_response(analysis)
