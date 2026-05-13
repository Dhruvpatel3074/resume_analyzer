"""
ats_scorer.py
==============
ATS (Applicant Tracking System) scoring engine.

Responsibilities
----------------
- Orchestrate the full NLP pipeline: extract text → analyse keywords → match JD
- Compute a composite ATS score (0–100) with weighted sub-scores
- Detect resume sections (Education, Experience, Skills, Projects, Certifications)
- Generate prioritised improvement suggestions
- Return all results as a structured ATSResult with a `.to_dict()` / `.to_json()` method
- Support PDF and DOCX file extraction (optional; graceful fallback to raw text)

Dependencies (stdlib only for core logic):
  - keyword_analyzer.KeywordAnalyzer
  - jd_matcher.JDMatcher

Optional (for file extraction):
  - pypdf       → PDF text extraction
  - pdfplumber  → better PDF extraction (preferred over pypdf)
  - mammoth     → DOCX text extraction
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from resume_analyzer.nlp_pipeline.keyword_analyzer import KeywordAnalyzer
from resume_analyzer.nlp_pipeline.jd_matcher import JDMatcher, MatchReport


# ---------------------------------------------------------------------------
# Score weights (sub-scores → composite ATS score)
# ---------------------------------------------------------------------------
ATS_WEIGHTS: dict[str, float] = {
    "keyword_match":        0.30,
    "skill_relevance":      0.25,
    "experience_alignment": 0.20,
    "education_fit":        0.15,
    "formatting_clarity":   0.10,
}

# Section heading patterns (regex, case-insensitive)
SECTION_PATTERNS: dict[str, list[str]] = {
    "summary":        [r"summary", r"objective", r"profile", r"about me", r"overview"],
    "experience":     [r"experience", r"work history", r"employment", r"career", r"positions?"],
    "education":      [r"education", r"academic", r"qualification", r"degree", r"university"],
    "skills":         [r"skills?", r"technical skills?", r"competenc", r"technologies"],
    "projects":       [r"projects?", r"portfolio", r"work samples?", r"contributions?"],
    "certifications": [r"certif", r"licens", r"credential", r"accreditation"],
    "awards":         [r"awards?", r"honors?", r"achievements?", r"recognition"],
    "publications":   [r"publications?", r"research", r"papers?"],
}

# Formatting quality signals
POSITIVE_FORMAT_SIGNALS = [
    r'\b\d{4}\b',                     # Years (dates present)
    r'\b\d+[\+%]?\s*(years?|yrs?)\b', # Experience duration
    r'@',                              # Email
    r'\+?\d[\d\s\-\(\)]{7,}',         # Phone
    r'linkedin\.com|github\.com',     # Professional URLs
    r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b',  # Proper nouns (names, companies)
]
NEGATIVE_FORMAT_SIGNALS = [
    r'[^\x00-\x7F]{5,}',    # Long non-ASCII runs (encoding issues)
    r'(.)\1{6,}',            # Repeated characters (garbled text)
    r'\bpage \d+ of \d+\b',  # Page markers that slipped through
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SectionInfo:
    """Detected resume section with content summary."""
    name: str
    detected: bool
    content_preview: str = ""   # First 200 chars of section content
    line_count: int = 0


@dataclass
class ImprovementSuggestion:
    """A single prioritised improvement recommendation."""
    priority: str    # "high" | "medium" | "low"
    category: str    # "skills" | "keywords" | "formatting" | "certifications" | "content"
    suggestion: str


@dataclass
class ATSResult:
    """
    Master result object returned by ATSScorer.score().

    All sub-objects have .to_dict() methods; call .to_json() for the
    complete pipeline output as a JSON string.
    """
    ats_score: int                                         # 0–100 composite
    grade: str                                             # Excellent / Good / Fair / Poor
    grade_detail: str
    score_breakdown: dict[str, int]                        # per-dimension points
    sections: list[SectionInfo] = field(default_factory=list) # type: ignore
    technical_skills: list[str] = field(default_factory=list) # type: ignore
    certifications: list[str] = field(default_factory=list) # type: ignore
    keywords_required: list[dict] = field(default_factory=list) # type: ignore
    match_report: Optional[MatchReport] = None
    improvements: list[ImprovementSuggestion] = field(default_factory=list) # type: ignore
    resume_text_length: int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> dict: # type: ignore
        return {
            "ats_score": self.ats_score,
            "grade": self.grade,
            "grade_detail": self.grade_detail,
            "score_breakdown": self.score_breakdown,
            "resume_text_length": self.resume_text_length,
            "sections_detected": {
                s.name: {
                    "detected": s.detected,
                    "line_count": s.line_count,
                    "preview": s.content_preview,
                }
                for s in self.sections
            },
            "technical_skills": self.technical_skills,
            "certifications": self.certifications,
            "keywords_required": self.keywords_required, # type: ignore
            "match_analysis": self.match_report.to_dict() if self.match_report else None, # type: ignore
            "improvements": [
                {
                    "priority": i.priority,
                    "category": i.category,
                    "suggestion": i.suggestion,
                }
                for i in self.improvements
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full pipeline result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) # type: ignore

    def print_summary(self) -> None:
        """Pretty-print a console summary of the ATS result."""
        bar_len = 40
        filled = int(self.ats_score / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print("=" * 60)
        print(f"  ATS SCORE: {self.ats_score}/100  [{self.grade}]")
        print(f"  {bar}")
        print(f"  {self.grade_detail}")
        print("=" * 60)

        print("\n── Score Breakdown ──────────────────────────────────")
        for dim, pts in self.score_breakdown.items():
            print(f"  {dim:<25} {pts:>3} pts")

        print("\n── Sections Detected ────────────────────────────────")
        for s in self.sections:
            mark = "✓" if s.detected else "✗"
            print(f"  {mark}  {s.name.capitalize():<20} ({s.line_count} lines)")

        print("\n── Technical Skills ─────────────────────────────────")
        skills_line = ", ".join(self.technical_skills[:15]) or "None detected"
        print(textwrap.fill(f"  {skills_line}", width=58, subsequent_indent="  "))

        if self.certifications:
            print("\n── Certifications ───────────────────────────────────")
            print("  " + ", ".join(self.certifications))

        print("\n── Improvements ─────────────────────────────────────")
        for imp in self.improvements[:8]:
            icon = {"high": "⚠", "medium": "→", "low": "·"}[imp.priority]
            print(textwrap.fill(
                f"  {icon} [{imp.priority.upper()}] {imp.suggestion}",
                width=58, subsequent_indent="      ",
            ))
        print("=" * 60)


# ---------------------------------------------------------------------------
# Text extraction utilities
# ---------------------------------------------------------------------------
def extract_text_from_file(filepath: str | Path) -> str:
    """
    Extract raw text from a PDF or DOCX file.

    Parameters
    ----------
    filepath : Path to the resume file (.pdf, .docx, or .txt)

    Returns
    -------
    Extracted text string

    Raises
    ------
    ValueError : If the file type is unsupported
    ImportError : If required extraction library is not installed
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        return _extract_pdf(path)

    if suffix in (".docx", ".doc"):
        return _extract_docx(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")


def _extract_pdf(path: Path) -> str:
    """Try pdfplumber first, fall back to pypdf."""
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path.resolve()}")

    pdfplumber_err: Exception | None = None

    # Try pdfplumber (richer extraction)
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:  # type: ignore
            pages = [p.extract_text() or "" for p in pdf.pages]  # type: ignore
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception as e:
        pdfplumber_err = e

    # Fall back to pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))  # type: ignore
        return "\n".join(
            page.extract_text() or "" for page in reader.pages  # type: ignore
        ).strip()
    except ImportError as e:
        raise ImportError(
            "PDF extraction requires pdfplumber or pypdf.\n"
            "Install with: pip install pdfplumber pypdf"
        ) from e
    except Exception as e:
        if pdfplumber_err is not None:
            raise RuntimeError(
                f"PDF extraction failed with pdfplumber ({pdfplumber_err}) and with pypdf ({e})."
            ) from e
        raise


def _extract_docx(path: Path) -> str:
    try:
        import mammoth  # type: ignore
        with open(str(path), "rb") as f:
            result = mammoth.extract_raw_text(f)  # type: ignore
        return result.value.strip()  # type: ignore
    except ImportError:
        raise ImportError(
            "DOCX extraction requires mammoth.\n"
            "Install with: pip install mammoth"
        )


# ---------------------------------------------------------------------------
# ATS Scorer
# ---------------------------------------------------------------------------
class ATSScorer:
    """
    Full ATS scoring pipeline.

    Usage
    -----
    # From raw text
    >>> scorer = ATSScorer()
    >>> result = scorer.score(resume_text, jd_text)
    >>> print(result.ats_score)

    # From file
    >>> result = scorer.score_file("resume.pdf", jd_text)
    >>> print(result.to_json())
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or ATS_WEIGHTS
        self._validate_weights()
        self.analyzer = KeywordAnalyzer()
        self.matcher = JDMatcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, resume_text: str, jd_text: str) -> ATSResult:
        """
        Score a resume against a job description.

        Parameters
        ----------
        resume_text : Extracted resume text
        jd_text     : Job description text

        Returns
        -------
        ATSResult with composite score, breakdown, and suggestions
        """
        # Step 1 – Keyword & skill extraction
        resume_report = self.analyzer.analyze(resume_text, source="resume")

        # Step 2 – JD matching
        match = self.matcher.match(resume_text, jd_text)

        # Step 3 – Detect resume sections
        sections = self._detect_sections(resume_text)

        # Step 4 – Formatting quality
        fmt_score = self._assess_formatting(resume_text)

        # Step 5 – Compute sub-scores (each dimension 0–max)
        breakdown = self._compute_breakdown(match, sections, fmt_score)

        # Step 6 – Composite score
        total = sum(breakdown.values())
        ats_score = min(100, max(0, round(total)))

        grade, grade_detail = self._grade(ats_score)

        # Step 7 – Keyword matrix (for JSON output)
        kw_required = self._build_keyword_matrix(match) # type: ignore

        # Step 8 – Improvement suggestions
        suggestions = self._generate_suggestions( # type: ignore
            match, sections, fmt_score, resume_report
        )

        return ATSResult(
            ats_score=ats_score,
            grade=grade,
            grade_detail=grade_detail,
            score_breakdown=breakdown,
            sections=sections,
            technical_skills=resume_report.top_technical,
            certifications=resume_report.certifications_found,
            keywords_required=kw_required,
            match_report=match,
            improvements=suggestions,
            resume_text_length=len(resume_text),
        )

    def score_file(self, filepath: str | Path, jd_text: str) -> ATSResult:
        """
        Extract text from a PDF/DOCX file and score it.

        Parameters
        ----------
        filepath : Path to resume file
        jd_text  : Job description text
        """
        resume_text = extract_text_from_file(filepath)
        return self.score(resume_text, jd_text)

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------
    def _detect_sections(self, text: str) -> list[SectionInfo]:
        lines = text.split("\n")
        sections: list[SectionInfo] = []

        for section_name, patterns in SECTION_PATTERNS.items():
            combined = re.compile(
                r'^\s*(' + "|".join(patterns) + r')\s*[:\-–]?\s*$',
                re.IGNORECASE,
            )
            detected = False
            content_lines: list[str] = []
            capture = False

            for line in lines:
                if combined.match(line.strip()):
                    detected = True
                    capture = True
                    continue
                if capture:
                    # Stop at next section heading or blank block
                    if re.match(r'^\s*[A-Z][A-Z\s]{3,}\s*$', line) and len(line.strip()) < 40:
                        break
                    content_lines.append(line.strip())
                    if len(content_lines) >= 20:
                        break

            # Fallback: check if keywords appear anywhere
            if not detected:
                any_pattern = re.compile("|".join(patterns), re.IGNORECASE)
                if any_pattern.search(text):
                    detected = True

            preview = " ".join(content_lines[:3])[:200]
            sections.append(SectionInfo(
                name=section_name,
                detected=detected,
                content_preview=preview,
                line_count=len(content_lines),
            ))

        return sections

    # ------------------------------------------------------------------
    # Formatting assessment
    # ------------------------------------------------------------------
    def _assess_formatting(self, text: str) -> float:
        """Return a 0.0–1.0 formatting quality score."""
        score = 0.5  # baseline

        pos_hits = sum(
            1 for pat in POSITIVE_FORMAT_SIGNALS
            if re.search(pat, text)
        )
        neg_hits = sum(
            1 for pat in NEGATIVE_FORMAT_SIGNALS
            if re.search(pat, text, re.IGNORECASE)
        )

        score += pos_hits * 0.08
        score -= neg_hits * 0.15

        # Length sanity check
        word_count = len(text.split())
        if word_count < 100:
            score -= 0.3   # suspiciously short
        elif word_count > 2000:
            score -= 0.1   # possibly too verbose

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------
    def _compute_breakdown(
        self,
        match: MatchReport,
        sections: list[SectionInfo],
        fmt_score: float,
    ) -> dict[str, int]:
        """
        Map match report dimensions to ATS point buckets.

        Returns dict with integer points per dimension.
        """
        dim_map = {d.name: d.raw_score for d in match.dimensions}

        keyword_pts  = round(dim_map.get("keyword_coverage", 0)    * 30)
        skill_pts    = round(dim_map.get("skill_overlap", 0)        * 25)
        exp_pts      = round(dim_map.get("experience_signals", 0)   * 20)
        edu_pts      = round(dim_map.get("education_match", 0)      * 15)
        fmt_pts      = round(fmt_score                               * 10)

        return {
            "keyword_match":        keyword_pts,
            "skill_relevance":      skill_pts,
            "experience_alignment": exp_pts,
            "education_fit":        edu_pts,
            "formatting_clarity":   fmt_pts,
        }

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    @staticmethod
    def _grade(score: int) -> tuple[str, str]:
        if score >= 80:
            return "Excellent", "Strong match — resume is well-optimized for this role."
        if score >= 65:
            return "Good", "Good match with a few skill or keyword gaps to address."
        if score >= 45:
            return "Fair", "Moderate match — targeted improvements could raise your chances significantly."
        return "Poor", "Low match — consider tailoring your resume specifically for this job description."

    # ------------------------------------------------------------------
    # Keyword matrix builder
    # ------------------------------------------------------------------
    def _build_keyword_matrix(self, match: MatchReport) -> list[dict]: # type: ignore
        matrix: list[dict] = [] # type: ignore
        resume_set = set(match.matched_keywords) | set(match.matched_skills) # type: ignore

        for kd in match.keyword_details:
            matrix.append({ # type: ignore
                "keyword": kd.keyword,
                "found": kd.in_resume,
                "importance": kd.importance,
            })

        # Add missing skills not already in matrix
        existing = {m["keyword"] for m in matrix} # type: ignore
        for skill in match.missing_skills:
            if skill not in existing:
                matrix.append({"keyword": skill, "found": False, "importance": "required"}) # type: ignore

        return matrix # type: ignore

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------
    def _generate_suggestions(
        self,
        match: MatchReport,
        sections: list[SectionInfo],
        fmt_score: float,
        resume_report, # type: ignore
    ) -> list[ImprovementSuggestion]:
        suggestions: list[ImprovementSuggestion] = []

        # --- Missing critical skills ---
        if match.missing_skills:
            top_missing = ", ".join(match.missing_skills[:5])
            suggestions.append(ImprovementSuggestion(
                priority="high",
                category="skills",
                suggestion=(
                    f"Add or highlight these skills missing from your resume: {top_missing}. "
                    "Include them in your Skills section and weave them into experience bullet points."
                ),
            ))

        # --- Low keyword coverage ---
        kw_dim = next((d for d in match.dimensions if d.name == "keyword_coverage"), None)
        if kw_dim and kw_dim.raw_score < 0.6:
            missing_kws = ", ".join(match.missing_keywords[:6])
            suggestions.append(ImprovementSuggestion(
                priority="high",
                category="keywords",
                suggestion=(
                    f"Your resume covers only {kw_dim.raw_score * 100:.0f}% of the JD's key terms. "
                    f"Mirror the JD's exact language for: {missing_kws}."
                ),
            ))

        # --- Missing sections ---
        section_map = {s.name: s for s in sections}
        critical = ["experience", "education", "skills"]
        for sec in critical:
            if not section_map.get(sec, SectionInfo(sec, False)).detected:
                suggestions.append(ImprovementSuggestion(
                    priority="high",
                    category="content",
                    suggestion=(
                        f"No '{sec.capitalize()}' section detected. ATS systems parse this section "
                        "explicitly — add a clearly-labelled heading."
                    ),
                ))

        # --- Missing certifications ---
        if match.missing_certs:
            cert_str = ", ".join(match.missing_certs[:3])
            suggestions.append(ImprovementSuggestion(
                priority="medium",
                category="certifications",
                suggestion=(
                    f"The JD references certifications not found in your resume: {cert_str}. "
                    "Pursue or list any equivalent credentials you hold."
                ),
            ))

        # --- Experience gap ---
        exp_dim = next((d for d in match.dimensions if d.name == "experience_signals"), None)
        if exp_dim and exp_dim.raw_score < 0.7:
            details = exp_dim.details # type: ignore
            suggestions.append(ImprovementSuggestion(
                priority="medium",
                category="content",
                suggestion=(
                    f"JD requires ~{details.get('jd_years_required', '?')} years; " # type: ignore
                    f"your resume signals ~{details.get('resume_years_detected', '?')}. " # type: ignore
                    "Include explicit date ranges (MM/YYYY – MM/YYYY) for all roles."
                ),
            ))

        # --- Formatting issues ---
        if fmt_score < 0.6:
            suggestions.append(ImprovementSuggestion(
                priority="medium",
                category="formatting",
                suggestion=(
                    "Formatting quality score is low. Ensure your resume includes: "
                    "contact email, phone number, LinkedIn/GitHub URL, and consistent date formats."
                ),
            ))

        # --- No summary section ---
        if not section_map.get("summary", SectionInfo("summary", False)).detected:
            suggestions.append(ImprovementSuggestion(
                priority="low",
                category="content",
                suggestion=(
                    "Add a 3-line professional summary at the top of your resume. "
                    "Incorporate the JD's top keywords to pass early ATS filtering stages."
                ),
            ))

        # --- Bonus skills to promote ---
        if match.bonus_skills:
            bonus_str = ", ".join(match.bonus_skills[:4])
            suggestions.append(ImprovementSuggestion(
                priority="low",
                category="skills",
                suggestion=(
                    f"You have skills not mentioned in the JD ({bonus_str}). "
                    "Keep them but ensure JD-required skills are more prominently featured."
                ),
            ))

        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: order[s.priority])
        return suggestions

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"ATS_WEIGHTS must sum to 1.0, got {total:.3f}."
            )

