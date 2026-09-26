# Research Assistant API

FastAPI backend for the research file management and AI-assisted analysis platform.

All routes except `/health` are served under the `/api/v1` prefix and follow
`../API_CONTRACT.md`, which is the specification the ResearchMind frontend is
built against. Field names are camelCase, timestamps are ISO 8601 UTC with
milliseconds and a `Z` suffix, and ids are url-safe strings.

## Current Phase

Phase 1 implements local SQLite persistence and JWT authentication:

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET /health`

Phase 2 adds project management:

- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `DELETE /api/v1/projects/{project_id}`

Phase 3 adds project-scoped document uploads:

- `POST /api/v1/projects/{project_id}/documents`
- `GET /api/v1/projects/{project_id}/documents`
- `DELETE /api/v1/projects/{project_id}/documents/{document_id}`

Every project and document endpoint requires a bearer token and is scoped to the
authenticated user. Document extraction, transcription, embeddings, RAG, chat,
reports, billing, and the assistant are not implemented yet.

## Local Setup

From the `backend` directory:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and replace the development JWT secret:

```bash
cp .env.example .env
```

The default development database URL is:

```env
DATABASE_URL=sqlite:///./data/app.db
```

The `data/` directory is created automatically. In development the app runs the Alembic migrations on startup, so the tables exist before the first request (see Database Migrations).

Start the API:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger or `http://127.0.0.1:8000/health` for the health response.

## Authentication

Registration requires a name, email, and password, and accepts an optional
`organization`. Passwords are hashed with Argon2. Login and registration both
return the session envelope from `API_CONTRACT.md` section 2.1. Send the token to
protected endpoints:

```text
Authorization: Bearer <accessToken>
```

```json
{
  "user": {
    "id": "5afeb148-09b4-41c4-af18-f6d8cd88f4e0",
    "name": "Dr. T. Moyo",
    "email": "t.moyo@university.ac.zw",
    "role": "researcher",
    "title": null,
    "organization": "University of Zimbabwe",
    "avatarUrl": null,
    "createdAt": "2026-01-08T11:20:00.000Z"
  },
  "accessToken": "eyJhbGciOi...",
  "expiresIn": 900
}
```

Access tokens last `ACCESS_TOKEN_EXPIRE_MINUTES` (15 by default), and
`expiresIn` is that value in seconds. `GET /api/v1/auth/me` returns the `user`
object on its own, for session restore.

`role`, `title`, `organization`, and `avatarUrl` are stored but not yet writable
through an endpoint; `role` defaults to `researcher` on registration.

### Logout

`POST /api/v1/auth/logout` returns `204` and revokes the bearer token it was
called with. Every access token carries a unique `jti` claim; logout stores it in
the `revoked_tokens` table, and every authenticated route rejects a revoked token
with `401 {"detail": "Token has been revoked"}`. Only that token is affected, so
the user's other sessions stay signed in. Revocation rows are needed only until
the token's own expiry, and expired rows are purged on each logout. Tokens without
a `jti` (issued before logout existed) are rejected.

## Projects

Every project endpoint requires `Authorization: Bearer <accessToken>` and only
ever exposes projects owned by the authenticated user.

`POST /api/v1/projects` accepts a `name` and an optional `description`. Both
values are trimmed, a blank or over-200-character name is rejected with `422`,
and a name the caller already used is rejected with `409`.

```text
POST /api/v1/projects
Authorization: Bearer <accessToken>
Content-Type: application/json

{
  "name": "Healthcare Access Study",
  "description": "Understanding the barriers that affect access to primary healthcare"
}
```

The response is `201` with the full project shape from `API_CONTRACT.md`
section 3.1:

```json
{
  "id": "c9e94b75-6dc9-4e32-97f5-8c357a968545",
  "name": "Healthcare Access Study",
  "description": "Understanding the barriers that affect access to primary healthcare",
  "documentCount": 0,
  "participantCount": 0,
  "interviewCount": 0,
  "focusGroupCount": 0,
  "themes": [],
  "codes": [],
  "status": "active",
  "createdAt": "2026-05-12T08:00:00.000Z",
  "updatedAt": "2026-05-12T08:00:00.000Z"
}
```

Ownership comes from the token, never from the request body, so a project can
only be created for the authenticated user. Omitting `description` stores an
empty string.

`GET /api/v1/projects` returns a bare array of the caller's projects, newest
`updatedAt` first. A user with no projects gets `[]`.

`GET /api/v1/projects/{project_id}` returns a single project, or `404` with
`{"detail": "Project not found"}`.

`DELETE /api/v1/projects/{project_id}` returns `204` with an empty body. The
project, its document rows, and its `uploads/{project_id}/` folder are removed
permanently; there is no soft delete. Documents are deleted explicitly because
SQLite does not enforce `ON DELETE CASCADE` unless foreign keys are switched on.

### Denormalised counters

`documentCount`, `interviewCount`, and `focusGroupCount` come from one grouped
aggregate query over the project's documents, so the frontend never has to build
a project card from extra requests. The contract never defines what the last two
count, so they are derived from `Document.kind`: `interview` and `focus_group`.
That means both stay `0` until `kind` is derived from document content instead of
the extension, which is a contract requirement once the processing pipeline
exists. The extension-derived kinds `pdf`, `docx`, `csv`, `audio`, and `video`
are all valid `DocumentKind` values, so nothing is rejected for having one.

`participantCount`, `themes`, and `codes` are always `0` and `[]` because
participants, themes, and codes are not implemented yet. The fields are present
so the response shape already matches the contract.

`updatedAt` is bumped whenever a document is added to or removed from the
project, which is what drives the "updated 2 days ago" label on a project card.

### Ownership

Reads and deletes are filtered by `user_id` in SQL, so one user can never see or
remove another user's project. Both cases return `404` rather than `403`, which
avoids revealing that the id exists.

## Documents

Documents provide project context. Uploads are project-scoped and stored on the
local filesystem under `uploads/{project_id}/`.

`POST /api/v1/projects/{project_id}/documents` takes `multipart/form-data` with
one or more repeated `files` parts, and returns `202` with a bare array holding
one record per accepted file.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/projects/$PROJECT_ID/documents \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "files=@Interview_01.docx" -F "files=@Community_Meeting.mp4"
```

```json
[
  {
    "id": "10467109-2d23-43de-ba7a-cb91be5c7df6",
    "projectId": "c9e94b75-6dc9-4e32-97f5-8c357a968545",
    "name": "Interview_01.docx",
    "kind": "docx",
    "type": "Interview",
    "extension": "DOCX",
    "status": "processing",
    "participantCount": 0,
    "language": null,
    "uploadedAt": "2026-09-17T07:10:00.000Z",
    "updatedAt": "2026-09-17T07:10:00.000Z",
    "fileSizeBytes": 131072,
    "fileSize": "128 KB"
  }
]
```

`fileSize` is a pre-formatted display string because the frontend renders it
directly; `fileSizeBytes` is the real byte count. Formatting uses 1024-based
units with one decimal place below 10 and none above, so `131072` becomes
`128 KB` and `432013312` becomes `412 MB`.

`storage_path` is intentionally not exposed, so the internal filesystem layout
never leaks to clients.

### Extension mapping

The accepted set matches the frontend's `ACCEPTED_EXTENSIONS` filter. The server
re-validates and never trusts that list.

| Extension | `kind`    | `type`            | Default limit |
| --------- | --------- | ----------------- | ------------- |
| `.mp3`    | `audio`   | `Interview`       | 10 MB         |
| `.wav`    | `audio`   | `Interview`       | 10 MB         |
| `.mp4`    | `video`   | `Research Notes`  | 100 MB        |
| `.pdf`    | `pdf`     | `Research Notes`  | 5 MB          |
| `.docx`   | `docx`    | `Interview`       | 5 MB          |
| `.csv`    | `csv`     | `Research Notes`  | 5 MB          |
| `.txt`    | `notes`   | `Research Notes`  | 5 MB          |

`type` values must match the frontend's hard-coded filter chips exactly, because
filtering is a `===` comparison.

Limits are configured with `UPLOAD_MAX_AUDIO_SIZE_MB`,
`UPLOAD_MAX_VIDEO_SIZE_MB`, and `UPLOAD_MAX_DOCUMENT_SIZE_MB`, and are
independent: a 6 MB `.mp3` is accepted while a 6 MB `.pdf` is rejected.

Upload responses:

| Status | Meaning                                                     |
| ------ | ----------------------------------------------------------- |
| `202`  | Stored, `status` is `processing`                            |
| `400`  | Empty file                                                  |
| `401`  | Missing or invalid token                                    |
| `404`  | Project missing or not owned by the caller                  |
| `413`  | Over the limit for that file's kind                         |
| `415`  | Extension is not an accepted type                           |
| `422`  | No files in the batch                                       |

`status` is `processing` because the contract makes upload asynchronous; see
Document Processing below for what happens next.

A batch with some good and some bad files stores the good ones and returns
`202` with only those. When every file is rejected, the request fails with the
first file's error status. This is the partial-batch behaviour the contract
asks for; rejected files get no record.

The client-supplied filename is reduced to its basename, so a name like
`../../etc/passwd.pdf` is stored as `passwd.pdf`. The file itself is written
under a random UUID name, which stops two uploads of the same filename from
overwriting each other.

`GET /api/v1/projects/{project_id}/documents` returns a bare array of the
project's documents, newest `uploadedAt` first.

`DELETE /api/v1/projects/{project_id}/documents/{document_id}` returns `204` and
removes both the database row and the file from disk. The document must belong to
the project in the path and to the caller's project, otherwise `404`.

### Where the logic lives

- `app/api/documents.py` only translates HTTP: it hands `(filename, stream)`
  pairs to the service and turns an all-rejected batch into the first file's
  status code.
- `app/services/documents.py` validates, stores and records uploads, and lists
  and deletes documents. A failed database write removes the file it just saved.
- `app/services/storage.py` is the only code that touches files. It exposes a
  `Storage` interface (`save`, `read`, `exists`, `delete`, `delete_prefix`) with a
  `LocalStorage` implementation rooted at `uploads/`. `Document.storage_path` is a
  storage key such as `{project_id}/{uuid}.pdf`; keys that could escape the root
  (`..`, absolute paths, backslashes, drive letters) are refused. The Oracle VM
  move adds a second implementation returned by `get_storage()`.
- Ownership checks live in `app/core/dependencies.py` as `CurrentProject` and
  `CurrentDocument`.

## Document Processing

Every upload is turned into searchable text in the background:

```text
upload (status: processing)
  → read file from storage
  → extract text        app/services/document_processor.py
  → split into chunks   app/utils/chunking.py
  → status: processed   (or failed, with processingError)
```

- **Formats:** PDF (text per page; scanned or password-protected PDFs fail with
  a reason), DOCX (paragraphs and tables, in order), TXT (UTF-8, UTF-16 with a
  byte-order mark, or Windows-1252), CSV (one `Header: value | …` line per row).
  Audio and video stay `processing` until transcription is added.
- **Outputs:** the full text is a `results` row with
  `result_type = "extracted_text"`. Chunks go to `document_chunks` (about 1,000
  characters with about 150 characters of overlap, never spanning two PDF pages),
  each with its `project_id`, page and exact character offsets. The `embedding`
  column is filled in a later step.
- **Worker:** `app/services/processing_worker.py` runs a background thread that
  takes waiting documents oldest first. Uploads wake it immediately; otherwise it
  checks every 5 seconds. The work list is just "documents in `processing`", so
  nothing is lost on restart. Set `PROCESSING_WORKER_ENABLED=false` to turn it off.
  Tests drive it directly with `ProcessingWorker.run_pending()`.
- **Endpoints:**
  - `POST /api/v1/projects/{project_id}/documents/{document_id}/reprocess` →
    `202` with the document back in `processing`; `409` if it is already
    processing. Earlier output is replaced.
  - `GET /api/v1/projects/{project_id}/documents/{document_id}/text` →
    `{ documentId, text, characterCount, chunkCount }`; `409` until processed.
- Deleting a document or project removes its chunks and results through the
  database's `ON DELETE CASCADE`.

### Embeddings

After extraction, the same worker embeds chunks with BGE-M3
(`app/ai/embeddings.py`, `app/services/embedding.py`):

- The model (`EMBEDDING_MODEL`, default `BAAI/bge-m3`) is loaded with
  `sentence-transformers` on first use, not at startup. The first run downloads
  about 2.3 GB into the Hugging Face cache. On the development laptop's CPU it
  embeds about 0.3 s per chunk.
- Vectors are normalised 1024-dimension float32 values stored as bytes in
  `document_chunks.embedding`, with the model name in `embedding_model`. Chunks
  embedded by a different model are re-embedded automatically.
- Embedding is a separate stage: a document is `processed` once its text is
  chunked. If the model cannot load, documents are not failed; the chunks wait,
  and the worker retries after 60 seconds.
- `EMBEDDINGS_ENABLED=false` turns the stage off.
- `GET /api/v1/projects/{project_id}/documents/{document_id}/text` includes
  `embeddedChunkCount`.

### Search

`GET /api/v1/projects/{project_id}/search?q=…&limit=5` returns the passages of
**that project only** that best match the question:

```json
[
  {
    "documentId": "…",
    "documentName": "interview_01.txt",
    "chunkIndex": 0,
    "page": null,
    "text": "Farmer: Getting our maize to the market. The transport costs have doubled…",
    "score": 0.64
  }
]
```

- `app/services/retrieval.py` embeds the question and ranks the project's
  chunks by cosine similarity in NumPy. Only chunks embedded by the current
  model are searched. After the Neon move, pgvector replaces `_nearest_chunks`.
- Results scoring below `RETRIEVAL_MIN_SCORE` (0.40) are dropped, so an
  unrelated question returns `[]`. Calibrated with BGE-M3: on-topic questions
  scored 0.55–0.69, off-topic 0.28–0.38.
- `limit` is 1–20. `503` if the embedding model cannot be loaded; `404` for
  another user's project.

### Summaries

A document's AI summary is generated in the background by the local LLM
(`OLLAMA_MODEL`, default `qwen2.5:3b`):

- `POST /api/v1/projects/{project_id}/documents/{document_id}/summary` → `202`
  with `status: "processing"`. `409` if the document is not processed yet, or a
  summary is already being generated. A new request replaces the old summary.
- `GET` the same path → `{ documentId, status, content, error, model, createdAt,
  completedAt }`. `status` becomes `ready` (Markdown `content` with **Overview**
  and **Key points**) or `failed` (`error`, e.g. "Ollama is not running…").
  `404` if none has been requested.
- `app/ai/llm.py` is the only code that talks to Ollama (`ChatModel` interface,
  `OllamaChatModel`). `app/services/summarizer.py` makes one call for short
  documents; longer ones are summarised per ~6,000-character section and the
  notes combined (map → reduce). Prompts quote the document as data and forbid
  invented conclusions.
- A second background thread, `GenerationWorker`, runs LLM jobs, so slow
  generation never holds up document processing. Summaries are stored in
  `results` (`result_type = "summary"`, with `status`, `error`, `model_name`,
  `completed_at`).
- Measured with `qwen2.5:3b` on the development laptop: 8–17 seconds for
  documents up to ~6,600 characters.

### Conversations

Chats with the project assistant are stored per project (`conversations`,
`messages`; every message also carries `project_id`):

- `GET /api/v1/projects/{project_id}/conversations` → newest activity first,
  each `{ id, projectId, title, preview, messageCount, createdAt, updatedAt }`.
  `preview` is the first question.
- `POST …/conversations` (optional `title`) → `201`. Untitled chats are named
  after their first question.
- `PATCH …/conversations/{conversation_id}` (`title`) renames;
  `DELETE` → `204` and removes its messages.
- `GET …/conversations/{conversation_id}/messages` → oldest first, each
  `{ id, role, content, createdAt, projectId, conversationId, evidence }`;
  `evidence` is the contract's `Evidence[]` on assistant answers.
- `GET /api/v1/projects/{project_id}/messages` → the most recent conversation's
  messages (contract §7.3).
- A conversation is only reachable through its own project (`404` otherwise).

### Assistant (RAG)

`POST /api/v1/projects/{project_id}/chat` with `{ "question": "…",
"conversationId": "…" }` (`conversationId` optional; omit it to start a new chat)
returns the assistant's message:

```json
{
  "id": "…",
  "role": "assistant",
  "content": "- P03 said taking maize to market now costs 10 dollars a bag, double last year's 5 dollars. [1]",
  "conversationId": "…",
  "projectId": "…",
  "createdAt": "2026-09-26T10:00:00.000Z",
  "evidence": [
    {
      "id": "<chunk id>",
      "source": "Interview 03",
      "participant": "P03",
      "quote": "Taking maize to the market in town now costs 10 dollars a bag, double last year's 5 dollars.",
      "documentId": "…",
      "relevance": 0.56
    }
  ]
}
```

How `app/services/assistant.py` answers:

0. Route the message (`needs_retrieval`): short small talk made only of known
   phrases ("hello", "thanks!", "how are you?") gets a brief conversational reply
   from the LLM, with the conversation history but no search and `evidence: []`.
   Anything else, including when in doubt, goes on to retrieval.
1. Search **this project's** chunks (up to 6 above `RETRIEVAL_MIN_SCORE`). Short
   follow-ups ("and fertiliser?") are searched together with the previous question.
2. If the project has no processed documents, or nothing relevant is found,
   reply with a plain "couldn't find anything" message and `evidence: []`,
   without calling the LLM.
3. Otherwise send the passages to the LLM as numbered `<source>` blocks, with the
   last two exchanges for context. The system prompt allows only the sources,
   forbids `[n]` citation markers in the text (attribution is the evidence's
   job), keeps figures with what they describe, forbids invented quotes and
   guessed gender, and treats sources as data, not instructions.
4. The top three passages become `Evidence`: a **verbatim** quote (the
   best-matching run of sentences, 80–320 characters, cut straight from the
   passage), the participant label if one appears (else `—`), page, relevance
   and document id. If the model writes `[n]` markers anyway, the cited sources
   are used instead, and the markers are removed from the answer (also while
   streaming).
5. The question and answer are saved together only after a successful answer.

Errors: `404` conversation not in this project, `422` empty or over-2,000-character
question, `503` Ollama or the embedding model unavailable, `502` other model
errors. Measured on the development laptop: 3–7 seconds per answer.

#### Streaming

`POST /api/v1/projects/{project_id}/chat/stream` takes the same body and answers
with Server-Sent Events (`text/event-stream`) while the model writes:

```text
event: token
data: {"text": "- P03 planted"}

event: token
data: {"text": " less because"}

event: done
data: {"message": { …the saved ChatMessage, with evidence and conversationId… }}
```

If generation fails midway, the last event is `error` (`{"detail": "…"}`) and
nothing is saved. Problems found before the first word (unknown conversation,
search or Ollama unavailable) are ordinary HTTP errors, as for `/chat`. The
answer is saved in a fresh database session at the end, because the request's
session may already be closed while the response streams. With the model loaded,
the first words arrive after about 2.6 seconds.

### Project analysis

`POST /api/v1/projects/{project_id}/analyze` → `202`; the analysis of **all
processed documents** is written in the background (`409` if one is already
running or the project has no processed documents). `GET …/analysis` returns it:

```json
{
  "projectId": "…",
  "status": "ready",
  "content": "## Major themes\n- **Water supply** (Focus Group 01) …\n## Common findings\n…",
  "progress": null,
  "documents": [{ "id": "…", "name": "Interview_03.txt" }, …],
  "model": "qwen2.5:3b",
  "createdAt": "…",
  "completedAt": "…"
}
```

While it runs, `status` is `processing` and `progress` says what it is doing,
e.g. `"Reading documents (2 of 3): Interview_07.txt"` or `"Writing the analysis"`.

`app/services/analyzer.py` gets notes on each processed document (its ready
summary if it has one, otherwise a fresh summary of its text), condenses them if
together they are too long, then asks the LLM for six Markdown sections: Major
themes, Common findings, Differences, Patterns, Important observations and
Overall conclusions. Each point names the documents it comes from. The prompt
forbids "all documents" claims that the notes do not support, added
recommendations and guessed gender. One analysis per project: a re-run replaces
it. Measured: about 32 seconds for three documents on the development laptop.

### AI health

`GET /health/ai` (unprefixed, no login) reports the local AI stack without
loading anything:

```json
{
  "status": "ok",
  "llm": { "model": "qwen2.5:3b", "reachable": true, "modelAvailable": true, "error": null },
  "embeddings": { "model": "BAAI/bge-m3", "status": "ready", "enabled": true, "pendingChunks": 0 }
}
```

`status` is `degraded` when Ollama or the model is missing, or the embedder
failed. A blank `OLLAMA_MODEL` means `qwen2.5:3b`.

## Known Gaps Against the Contract

Deliberately not done yet, listed so they are not lost:

- No `{ "error": { "code", "message", "fieldErrors", "requestId" } }` envelope.
  Errors are still FastAPI's `{"detail": ...}`. The contract calls this a
  suggestion and states the frontend has no error handling, so any shape works.
- No `X-Request-Id`, `X-Idempotency-Key`, or `RateLimit-*` headers.
- No pagination. The contract asks for none on these routes, and the frontend has
  no "load more" control.
- Audio and video stay `processing` until transcription exists. There is no
  language detection, and no `PATCH` for user overrides of `kind` or `type`.
- No `PATCH /projects/{project_id}`, so `status` can never become `archived`.
- OAuth, password reset, and refresh tokens are not implemented.
- Plans, quotas, and billing are not implemented, so no `403 quota_exceeded`.

## Tests

The test suite runs against an isolated in-memory SQLite database and does not modify `data/app.db` or `uploads/`. The current count is 226 tests covering auth (including logout), projects, documents, storage, document processing, embeddings, search (including the project boundary), summaries, the Ollama client, conversations, the assistant (including streaming), project analysis, and migrations. A fake embedder and a fake LLM (`tests/conftest.py`) are installed for every test, so no test loads BGE-M3 or calls Ollama.

```bash
ruff check app alembic tests
python -m pytest -q
```

## Database Migrations

The schema is managed by Alembic. `alembic/versions/0001_baseline.py` creates
`users`, `projects`, `documents` and `revoked_tokens`.

- **Development:** the app applies migrations on startup (`app/db/migrations.py`).
  A database created by the old `create_all` startup (tables but no
  `alembic_version`) is stamped at `0001_baseline` first, then upgraded, so
  existing data is kept.
- **Other environments:** run `alembic upgrade head` before starting the app.

After changing a model:

```bash
alembic revision --autogenerate -m "describe the change"
# review the generated file in alembic/versions/, then
alembic upgrade head
```

SQLite notes:

- Foreign keys are switched on for every SQLite connection
  (`app/db/database.py`), so `ON DELETE CASCADE` works as it will on PostgreSQL.
- Migrations use batch mode (tables are copied to apply ALTERs), with foreign
  keys switched off for the duration so the copy cannot cascade-delete rows.
- Constraint names follow the naming convention in `app/db/base.py`.

`tests/test_migrations.py` checks that the migrations build exactly the schema
the models describe, so a model change without a migration fails the tests.

## Docker

```bash
docker build -t research-assistant-api .
docker run --env-file .env -p 8000:8000 research-assistant-api
```

The image does not bundle local AI models. Model installation and runtime configuration remain external to the container.

## Current Scope

The project contains the application boundaries for API routes, database models and sessions, schemas, services, AI integrations, configuration, and utilities. Authentication, the `User` model, project CRUD, and project-scoped document uploads are implemented locally with SQLite and the local filesystem, shaped to `../API_CONTRACT.md`. Neon, Oracle, Supabase Storage, document extraction, transcription, embeddings, vector search, RAG, Ollama integration, and assistant behavior remain future work.
