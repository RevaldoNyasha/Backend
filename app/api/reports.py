from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.core.dependencies import CurrentProject, CurrentUser, DbSession
from app.models.result import RESULT_STATUS_PROCESSING, RESULT_STATUS_READY, Result
from app.schemas.report import (
    ReportCreate,
    ReportGeneration,
    ReportResponse,
    ReportTemplateResponse,
    ReportUpdate,
)
from app.schemas.result import AnalysisDocument
from app.services.processing_worker import notify_generation_worker
from app.services.report_export import (
    DOCX_MEDIA_TYPE,
    PDF_MEDIA_TYPE,
    ReportDocument,
    filename_for,
    to_docx,
    to_pdf,
)
from app.services.report_templates import DEFAULT_TYPE, TEMPLATES
from app.services.reports import (
    REPORT_STATUS_DRAFT,
    ReportNotAvailable,
    delete_report,
    get_report,
    list_reports,
    request_report,
    update_report,
)

router = APIRouter(tags=["reports"])

REPORTS = "/projects/{project_id}/reports"
REPORT = REPORTS + "/{report_id}"


def to_response(report: Result, with_content: bool = False) -> ReportResponse:
    details = report.details or {}
    ready = report.status == RESULT_STATUS_READY
    return ReportResponse(
        id=report.id,
        project_id=report.project_id,
        title=details.get("title", "Report"),
        summary=details.get("summary", ""),
        created_at=report.created_at,
        sections=details.get("sections", []),
        status=details.get("reportStatus", REPORT_STATUS_DRAFT),
        type=details.get("type", DEFAULT_TYPE),
        documents=[AnalysisDocument(**document) for document in details.get("documents", [])],
        model=report.model_name,
        generation=ReportGeneration(
            state=report.status,
            progress=details.get("progress") if report.status == RESULT_STATUS_PROCESSING else None,
            error=report.error,
            started_at=report.created_at,
            completed_at=report.completed_at,
        ),
        content=report.content if with_content and ready else None,
    )


@router.get("/report-templates", response_model=list[ReportTemplateResponse])
def templates(_: CurrentUser) -> list[ReportTemplateResponse]:
    """The report types the system can produce, with their section outlines."""
    return [
        ReportTemplateResponse(
            type=template.type,
            label=template.label,
            description=template.description,
            sections=template.headings(),
            custom_sections=template.custom_sections,
        )
        for template in TEMPLATES.values()
    ]


@router.get(REPORTS, response_model=list[ReportResponse])
def index(project_id: str, current_project: CurrentProject, db: DbSession):
    """This project's reports, newest first (contract §8.2, a bare array)."""
    return [to_response(report) for report in list_reports(db, current_project)]


@router.post(REPORTS, response_model=ReportResponse, status_code=status.HTTP_202_ACCEPTED)
def create(project_id: str, payload: ReportCreate, current_project: CurrentProject, db: DbSession):
    """Start generating a report from a template; poll `GET …/reports/{id}` until ready."""
    try:
        report = request_report(
            db,
            current_project,
            report_type=payload.type,
            title=payload.title,
            sections=payload.sections,
            document_ids=payload.document_ids,
        )
    except ReportNotAvailable as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from None
    notify_generation_worker()
    return to_response(report)


@router.get(REPORT, response_model=ReportResponse)
def show(project_id: str, report_id: str, current_project: CurrentProject, db: DbSession):
    """One report with its Markdown `content` once generated (contract §8.4)."""
    return to_response(_report_or_404(db, current_project, report_id), with_content=True)


@router.patch(REPORT, response_model=ReportResponse)
def update(
    project_id: str,
    report_id: str,
    payload: ReportUpdate,
    current_project: CurrentProject,
    db: DbSession,
):
    """Rename a report, or mark it `final` / back to `draft`."""
    report = _report_or_404(db, current_project, report_id)
    try:
        report = update_report(db, report, title=payload.title, status=payload.status)
    except ReportNotAvailable as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None
    return to_response(report, with_content=True)


@router.delete(REPORT, status_code=status.HTTP_204_NO_CONTENT)
def destroy(project_id: str, report_id: str, current_project: CurrentProject, db: DbSession):
    delete_report(db, _report_or_404(db, current_project, report_id))


@router.get(REPORT + "/export")
def export(
    project_id: str,
    report_id: str,
    current_project: CurrentProject,
    db: DbSession,
    format: Annotated[Literal["pdf", "docx"], Query()] = "pdf",
) -> Response:
    """The report as a file (contract §8.5): the PDF or DOCX itself, not JSON."""
    report = _report_or_404(db, current_project, report_id)
    if report.status != RESULT_STATUS_READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This report is still being written. Export it once it is ready.",
        )
    details = report.details or {}
    template = TEMPLATES.get(details.get("type", ""), TEMPLATES[DEFAULT_TYPE])
    document = ReportDocument(
        title=details.get("title", "Report"),
        content=report.content,
        project_name=current_project.name,
        template_label=template.label,
        created_at=report.created_at,
        documents=[d["name"] for d in details.get("documents", [])],
    )
    if format == "docx":
        body, media_type = to_docx(document), DOCX_MEDIA_TYPE
    else:
        body, media_type = to_pdf(document), PDF_MEDIA_TYPE
    filename = filename_for(document, format)
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _report_or_404(db: DbSession, project, report_id: str) -> Result:
    report = get_report(db, project, report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report
