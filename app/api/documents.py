from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi import File as FileParam

from app.core.config import settings
from app.core.dependencies import CurrentDocument, CurrentProject, DbSession
from app.models.document import Document
from app.schemas.document import DocumentResponse, DocumentTextResponse, TranscriptResponse
from app.services.documents import delete_document, list_documents, upload_documents
from app.services.embedding import embedded_chunk_count
from app.services.processing import (
    DocumentAlreadyProcessing,
    TranscriptionDisabled,
    chunk_count_for,
    extracted_text_for,
    reprocess_document,
)
from app.services.processing_worker import notify_worker
from app.services.transcription import transcript_for

router = APIRouter(tags=["documents"])


@router.post(
    "/projects/{project_id}/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
def upload(
    project_id: str,
    files: Annotated[list[UploadFile], FileParam()],
    current_project: CurrentProject,
    db: DbSession,
) -> list[Document]:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files were uploaded",
        )

    result = upload_documents(
        db, current_project, ((upload_file.filename, upload_file.file) for upload_file in files)
    )

    # Partial batches succeed with the accepted files; an all-rejected batch
    # fails with the first file's reason.
    if not result.stored:
        first = result.rejected[0]
        raise HTTPException(status_code=first.status_code, detail=first.detail)

    return result.stored


@router.get("/projects/{project_id}/documents", response_model=list[DocumentResponse])
def index(project_id: str, current_project: CurrentProject, db: DbSession) -> list[Document]:
    return list_documents(db, current_project)


@router.post(
    "/projects/{project_id}/documents/{document_id}/reprocess",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> Document:
    try:
        document = reprocess_document(db, current_project, current_document)
    except DocumentAlreadyProcessing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already processing",
        ) from None
    except TranscriptionDisabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Audio transcription is turned off on this server",
        ) from None
    notify_worker()
    return document


@router.get(
    "/projects/{project_id}/documents/{document_id}/text",
    response_model=DocumentTextResponse,
)
def text(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> DocumentTextResponse:
    extracted = extracted_text_for(db, current_document)
    if extracted is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Text is not available until the document has been processed",
        )
    return DocumentTextResponse(
        document_id=current_document.id,
        text=extracted,
        character_count=len(extracted),
        chunk_count=chunk_count_for(db, current_document),
        embedded_chunk_count=embedded_chunk_count(
            db, current_document.id, settings.embedding_model
        ),
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/transcript",
    response_model=TranscriptResponse,
)
def transcript(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> TranscriptResponse:
    """The timed transcript of an audio or video document."""
    found = transcript_for(db, current_document)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="There is no transcript until the recording has been transcribed",
        )
    return found


@router.delete(
    "/projects/{project_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def destroy(
    project_id: str,
    document_id: str,
    current_project: CurrentProject,
    current_document: CurrentDocument,
    db: DbSession,
) -> None:
    delete_document(db, current_project, current_document)
