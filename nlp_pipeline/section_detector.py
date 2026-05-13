"""
section_detector.py
===================
Takes the clean text from extractor.py and splits it into
labelled sections: Summary, Experience, Education, Skills, etc.

HOW IT WORKS (3 strategies combined):
    1. Regex header matching  — catches "EXPERIENCE", "Work Experience", "experience:"
    2. spaCy sentence scoring — catches headers without obvious formatting
    3. Positional fallback    — if nothing found, makes a best guess by position
"""

import re
import spacy

# ── Load spaCy once at module level (expensive to load repeatedly) ──
nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────
# SECTION LABEL MAP
# Each key is the canonical label we'll use in our JSON output.
# Each value is a list of regex patterns that match that section's header.
#
# WHY REGEX AND NOT JUST string.lower() == "education"?
#   Real resumes use wildly inconsistent headers:
#   "EDUCATIONAL BACKGROUND", "Academic Qualifications",
#   "Education & Training", "Schooling" — all mean the same thing.
#   Regex lets us catch all variations with one pattern.
# ─────────────────────────────────────────────────────────────────────

SECTION_PATTERNS = {
    "summary": [
        r"(professional\s+)?summary",
        r"objective",
        r"profile",
        r"about\s+me",
        r"career\s+objective",
        r"professional\s+profile",
    ],
    "experience": [
        r"(work\s+)?experience",
        r"employment(\s+history)?",
        r"work\s+history",
        r"professional\s+experience",
        r"internship(s)?",
        r"positions?\s+held",
        r"career\s+history",
    ],
    "education": [
        r"education(\s+&\s+training)?",
        r"educational\s+background",
        r"academic(s)?(\s+background)?",
        r"qualifications?",
        r"schooling",
        r"degrees?",
    ],
    "skills": [
        r"(technical\s+)?skills?",
        r"core\s+competencies",
        r"competencies",
        r"technologies",
        r"tech\s+stack",
        r"tools?\s+&\s+technologies",
        r"programming\s+languages?",
        r"expertise",
    ],
    "projects": [
        r"projects?",
        r"personal\s+projects?",
        r"academic\s+projects?",
        r"key\s+projects?",
        r"portfolio",
    ],
    "certifications": [
        r"certifications?",
        r"certificates?",
        r"licenses?\s+&\s+certifications?",
        r"professional\s+development",
        r"courses?(\s+&\s+certifications?)?",
    ],
    "achievements": [
        r"achievements?",
        r"accomplishments?",
        r"awards?(\s+&\s+honors?)?",
        r"honors?",
        r"recognitions?",
    ],
    "languages": [
        r"languages?",
        r"language\s+proficiency",
    ],
    "volunteer": [
        r"volunteer(ing)?(\s+experience)?",
        r"community\s+service",
        r"social\s+work",
    ],
    "references": [
        r"references?",
        r"referees?",
    ],
}

# Pre-compile all patterns for speed
# WHY PRE-COMPILE?
#   re.compile() builds a finite automaton from the pattern.
#   If we compiled inside the loop, we'd rebuild it for every line
#   of every resume. Pre-compiling once = much faster.

_COMPILED_PATTERNS = {
    label: [
        re.compile(r"^\s*" + pat + r"\s*:?\s*$", re.IGNORECASE)
        for pat in patterns
    ]
    for label, patterns in SECTION_PATTERNS.items()
}


# ─────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────

def detect_sections(text: str) -> dict:
    """
    Splits resume text into labelled sections.

    Returns:
    {
        "summary":          "Motivated CS student with 2 years...",
        "experience":       "Software Intern at XYZ\nJune 2024...",
        "education":        "B.Tech Computer Science\nGCET 2025",
        "skills":           "Python, Django, PostgreSQL, Docker",
        "projects":         "Resume Analyzer — built NLP pipeline",
        "certifications":   "AWS Cloud Practitioner",
        "other":            "any text not matched to a section"
    }

    All values are strings (raw text of that section).
    skill_extractor.py will further parse the skills string.
    """
    lines = text.splitlines()

    # Step 1: Find all header lines and their positions
    header_positions = _find_headers(lines)

    # Step 2: If no headers found at all, try a looser match
    if not header_positions:
        header_positions = _find_headers_loose(lines)

    # Step 3: Slice text between headers into sections
    sections = _slice_sections(lines, header_positions)

    # Step 4: Extract contact info from the top of the resume
    # (before the first section header)
    sections["_header_block"] = _extract_header_block(lines, header_positions)

    return sections


# ─────────────────────────────────────────────────────────────────────
# HEADER DETECTION
# ─────────────────────────────────────────────────────────────────────

def _find_headers(lines: list) -> list:
    """
    Scans each line and checks if it matches a section header pattern.

    Returns a list of (line_index, label) tuples, sorted by position.

    Example: [(3, "summary"), (12, "experience"), (28, "education")]

    WHY CHECK LINE LENGTH?
        A section header is almost always a short line — usually
        1-4 words. A line like "I have experience in Python and Django"
        is NOT a header even though it contains "experience".
        Capping at 60 chars filters out content lines.
    """
    found = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Headers are short (not a full sentence)
        if len(stripped) == 0 or len(stripped) > 60:
            continue

        for label, compiled_list in _COMPILED_PATTERNS.items():
            for pattern in compiled_list:
                if pattern.match(stripped):
                    found.append((i, label))
                    break  # Don't double-label the same line

    return found


def _find_headers_loose(lines: list) -> list:
    """
    Fallback: looks for ALL-CAPS short lines as likely headers.

    WHY THIS FALLBACK?
        Some resumes use custom headers like "ACHIEVEMENTS & IMPACT"
        or "MY JOURNEY" that don't match our patterns. But headers
        are almost always:
          - ALL CAPS, OR
          - Title Case with no punctuation
          - Short (under 40 chars)

    This catches those cases and labels them "other" if we can't
    map them to a known section.
    """
    found = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) == 0 or len(stripped) > 40:
            continue

        # All caps line with at least 3 chars = likely a header
        if stripped.isupper() and len(stripped) >= 3:
            # Try to map to known label, else use "other"
            label = _map_to_label(stripped.lower())
            found.append((i, label))

    return found


def _map_to_label(text: str) -> str:
    """Maps a lowercased header string to a known label, or 'other'."""
    for label, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return label
    return "other"


# ─────────────────────────────────────────────────────────────────────
# SECTION SLICING
# ─────────────────────────────────────────────────────────────────────

def _slice_sections(lines: list, header_positions: list) -> dict:
    """
    Given header positions [(3,'summary'), (12,'experience'), ...],
    slice the lines array between consecutive headers.

    VISUAL EXAMPLE:
        Line 0:  "John Doe"
        Line 1:  "john@email.com"
        Line 2:  ""
        Line 3:  "SUMMARY"          ← header at index 3
        Line 4:  "Motivated CS..."
        ...
        Line 12: "EXPERIENCE"       ← header at index 12
        Line 13: "Software Intern"
        ...

        summary section  = lines[4:12]
        experience section = lines[13:next_header]
    """
    sections = {}

    if not header_positions:
        sections["other"] = "\n".join(lines)
        return sections

    for idx, (line_num, label) in enumerate(header_positions):
        # Content starts on the line AFTER the header
        start = line_num + 1

        # Content ends at the next header (or end of file)
        if idx + 1 < len(header_positions):
            end = header_positions[idx + 1][0]
        else:
            end = len(lines)

        content_lines = [l.strip() for l in lines[start:end] if l.strip()]
        content = "\n".join(content_lines)

        # If the same label appears twice (e.g. two "Skills" headers),
        # concatenate rather than overwrite
        if label in sections:
            sections[label] = sections[label] + "\n" + content
        else:
            sections[label] = content

    return sections


def _extract_header_block(lines: list, header_positions: list) -> str:
    """
    Extracts everything BEFORE the first section header.
    This usually contains: name, email, phone, LinkedIn, location.

    We store this as '_header_block' — the candidate info extractor
    (in pipeline.py) will parse name/email/phone from it.
    """
    if not header_positions:
        # No headers found — take first 10 lines as contact info
        return "\n".join(lines[:10])

    first_header_line = header_positions[0][0]
    block = [l.strip() for l in lines[:first_header_line] if l.strip()]
    return "\n".join(block)


# ─────────────────────────────────────────────────────────────────────
# QUICK TEST
# Usage: python section_detector.py
# (uses a hardcoded sample — replace with your own resume text)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
John Doe
john.doe@email.com | +91-9876543210 | LinkedIn: linkedin.com/in/johndoe
Ahmedabad, Gujarat

SUMMARY
Final year B.Tech Computer Science student with experience in
Python, Django, and REST API development. Passionate about ML.

EXPERIENCE
Software Development Intern — TechCorp, Ahmedabad
June 2024 – August 2024
- Built REST APIs using FastAPI and PostgreSQL
- Reduced query time by 40% through indexing

EDUCATION
B.Tech Computer Science Engineering
GCET, Vallabh Vidyanagar — 2021–2025  CGPA: 8.4

SKILLS
Python, Django, FastAPI, PostgreSQL, MySQL
Git, Docker, Linux, VS Code
Machine Learning, NLP, scikit-learn

PROJECTS
Resume Analyzer — NLP-based resume scoring tool
Built an NLP pipeline using spaCy and sentence-transformers

CERTIFICATIONS
AWS Cloud Practitioner — Amazon Web Services, 2024
Python for Data Science — Coursera, 2023
"""

    result = detect_sections(sample)

    print("=" * 50)
    for section, content in result.items():
        if content.strip():
            print(f"\n[{section.upper()}]")
            print(content[:200])  # Show first 200 chars of each
    print("=" * 50)