"""The report types ResearchMind can produce (API_CONTRACT.md §8.3 `type`).

Each template is a fixed outline: the section headings, in order, and what the
model should write under each. Edit this file to change what a report contains;
the frontend reads the list from `GET /report-templates`, so it follows along.

`custom` has no fixed outline: the researcher supplies the section headings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    heading: str
    guidance: str


@dataclass(frozen=True)
class ReportTemplate:
    type: str
    label: str
    description: str
    sections: tuple[Section, ...]
    # True for `custom`: the request supplies the headings.
    custom_sections: bool = False

    def headings(self, custom: list[str] | None = None) -> list[str]:
        return list(custom or []) if self.custom_sections else [s.heading for s in self.sections]


TEMPLATES: dict[str, ReportTemplate] = {
    template.type: template
    for template in (
        ReportTemplate(
            type="thematic_analysis",
            label="Thematic Analysis",
            description="The main themes across the documents, with the evidence behind each.",
            sections=(
                Section(
                    "Executive Summary",
                    "Two or three sentences: what the documents are and the main themes found.",
                ),
                Section(
                    "Methodology",
                    "Which documents were analysed and what kind of material they are (e.g. "
                    "interviews, course outlines). Say the themes were identified by reading "
                    "the documents' content; do not invent a method, sample size or dates.",
                ),
                Section(
                    "Major Themes",
                    "3-6 themes. For each: the theme name in bold, one or two sentences, and the "
                    "documents it appears in, in parentheses.",
                ),
                Section(
                    "Evidence",
                    "For the most important themes, a short phrase or fact from the notes that "
                    "supports it, with the document it comes from. Quote only words that appear "
                    "in the notes.",
                ),
                Section("Conclusions", "Two to four sentences drawing the themes together."),
            ),
        ),
        ReportTemplate(
            type="evidence_summary",
            label="Evidence Summary",
            description="The key findings and exactly which documents support each one.",
            sections=(
                Section("Overview", "Two or three sentences on what the documents cover."),
                Section(
                    "Key Findings",
                    "Bullet points: each finding in one sentence, followed by the documents that "
                    "support it in parentheses.",
                ),
                Section(
                    "Supporting Evidence",
                    "For each key finding, the specific facts, figures or short phrases from the "
                    "notes that support it, naming the document. Quote only words in the notes.",
                ),
                Section(
                    "Gaps and Limitations",
                    "What the documents do not cover, or where the evidence is thin or comes "
                    "from a single document.",
                ),
            ),
        ),
        ReportTemplate(
            type="comparison",
            label="Document Comparison",
            description="How the documents agree and differ, side by side.",
            sections=(
                Section(
                    "Overview",
                    "Each document in one sentence: what it is and what it focuses on.",
                ),
                Section(
                    "Similarities",
                    "Bullet points on what the documents have in common, naming the documents.",
                ),
                Section(
                    "Differences",
                    "Bullet points on where the documents differ in focus, content or view, "
                    "naming the documents. With only one document, say differences cannot be "
                    "assessed.",
                ),
                Section("Conclusions", "Two to four sentences on what the comparison shows."),
            ),
        ),
        ReportTemplate(
            type="custom",
            label="Custom Report",
            description="Your own title and section headings, filled from the documents.",
            sections=(),
            custom_sections=True,
        ),
    )
}

DEFAULT_TYPE = "thematic_analysis"
MAX_CUSTOM_SECTIONS = 10
