"""
jd_matcher.py
==============
NLP module for matching a resume against a job description and producing
a detailed gap analysis with semantic similarity scoring.

Responsibilities
----------------
- Parse and normalise both resume and JD text
- Compute overlap scores for skills, keywords, and experience signals
- Perform section-level alignment (Education, Experience, Skills, etc.)
- Identify matched keywords, missing keywords, and bonus extras
- Return a structured MatchReport with per-dimension scores

Dependencies: re, collections (stdlib)
             keyword_analyzer.KeywordAnalyzer  (same package)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from nlp_pipeline.keyword_analyzer import KeywordAnalyzer, KeywordReport


# ---------------------------------------------------------------------------
# Configurable weights (must sum to 1.0)
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS: dict[str, float] = {
    "skill_overlap":       0.35,   # technical skill match
    "keyword_coverage":    0.25,   # general keyword coverage
    "experience_signals":  0.20,   # years / seniority signals
    "education_match":     0.10,   # degree / field signals
    "certification_match": 0.10,   # cert alignment
}

# Seniority level signals in job descriptions
SENIORITY_PATTERNS: dict[str, list[str]] = {
    "junior":  ["junior", "entry", "entry-level", "0-2 years", "1 year", "graduate", "intern"],
    "mid":     ["mid", "mid-level", "2-4 years", "3 years", "associate"],
    "senior":  ["senior", "sr.", "5+ years", "5 years", "lead", "principal", "staff"],
    "manager": ["manager", "head of", "director", "vp", "vice president", "c-level"],
}

DEGREE_SIGNALS: dict[str, int] = {
    "phd": 4, "ph.d": 4, "doctorate": 4,
    "master": 3, "msc": 3, "mba": 3, "m.s": 3,
    "bachelor": 2, "bsc": 2, "b.s": 2, "b.e": 2, "b.tech": 2,
    "diploma": 1, "associate": 1,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class DimensionScore:
    """Score for one matching dimension."""
    name: str
    raw_score: float          # 0.0 – 1.0
    weight: float
    weighted_score: float     # raw_score * weight
    details: dict = field(default_factory=dict) # type: ignore


@dataclass
class KeywordMatchDetail:
    """Per-keyword match result."""
    keyword: str
    in_resume: bool
    in_jd: bool
    importance: str           # "required" | "preferred" | "bonus"


@dataclass
class MatchReport:
    """Complete resume ↔ JD match analysis."""
    overall_match_pct: float                          # 0–100
    dimensions: list[DimensionScore] = field(default_factory=list) # type: ignore
    matched_skills: list[str] = field(default_factory=list) # type: ignore
    missing_skills: list[str] = field(default_factory=list) # type: ignore
    bonus_skills: list[str] = field(default_factory=list)   # type: ignore # resume has, JD didn't ask
    matched_keywords: list[str] = field(default_factory=list) # type: ignore
    missing_keywords: list[str] = field(default_factory=list) # type: ignore
    matched_certs: list[str] = field(default_factory=list) # type: ignore
    missing_certs: list[str] = field(default_factory=list) # type: ignore
    jd_seniority: str = "unspecified"
    jd_required_degree: str = "unspecified"
    keyword_details: list[KeywordMatchDetail] = field(default_factory=list) # type: ignore

    def to_dict(self) -> dict: # type: ignore
        return {
            "overall_match_pct": round(self.overall_match_pct, 2),
            "jd_seniority_level": self.jd_seniority,
            "jd_required_degree": self.jd_required_degree,
            "dimensions": [
                {
                    "name": d.name,
                    "raw_score": round(d.raw_score, 4),
                    "weight": d.weight,
                    "weighted_score": round(d.weighted_score, 4),
                    "details": d.details, # type: ignore
                }
                for d in self.dimensions
            ],
            "skills": {
                "matched": self.matched_skills,
                "missing": self.missing_skills,
                "bonus_in_resume": self.bonus_skills,
            },
            "keywords": {
                "matched": self.matched_keywords,
                "missing": self.missing_keywords,
            },
            "certifications": {
                "matched": self.matched_certs,
                "missing": self.missing_certs,
            },
            "keyword_details": [
                {
                    "keyword": kd.keyword,
                    "in_resume": kd.in_resume,
                    "in_jd": kd.in_jd,
                    "importance": kd.importance,
                }
                for kd in self.keyword_details
            ],
        }


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
class JDMatcher:
    """
    Compares a resume against a job description across multiple NLP dimensions.

    Usage
    -----
    >>> matcher = JDMatcher()
    >>> report = matcher.match(resume_text, jd_text)
    >>> print(f"Match: {report.overall_match_pct:.1f}%")
    >>> print("Missing skills:", report.missing_skills)
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or DIMENSION_WEIGHTS
        self._validate_weights()
        self.analyzer = KeywordAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def match(self, resume_text: str, jd_text: str) -> MatchReport:
        """
        Full match pipeline.

        Parameters
        ----------
        resume_text : Extracted resume text
        jd_text     : Job description text

        Returns
        -------
        MatchReport with overall score and per-dimension breakdown
        """
        resume_report = self.analyzer.analyze(resume_text, source="resume")
        jd_report = self.analyzer.analyze(jd_text, source="job_description")

        dimensions: list[DimensionScore] = []

        # 1. Skill overlap
        skill_dim = self._score_skill_overlap(resume_report, jd_report)
        dimensions.append(skill_dim)

        # 2. Keyword coverage
        kw_dim, kw_details = self._score_keyword_coverage(resume_report, jd_report, resume_text, jd_text)
        dimensions.append(kw_dim)

        # 3. Experience signals
        exp_dim = self._score_experience(resume_text, jd_text)
        dimensions.append(exp_dim)

        # 4. Education match
        edu_dim = self._score_education(resume_text, jd_text)
        dimensions.append(edu_dim)

        # 5. Certification match
        cert_dim, matched_certs, missing_certs = self._score_certifications(resume_text, jd_text)
        dimensions.append(cert_dim)

        overall = sum(d.weighted_score for d in dimensions) * 100

        # Skill lists
        resume_skills = set(resume_report.top_technical)
        jd_skills = set(jd_report.top_technical)
        matched_skills = sorted(resume_skills & jd_skills)
        missing_skills = sorted(jd_skills - resume_skills)
        bonus_skills = sorted(resume_skills - jd_skills)

        # Keyword lists
        resume_kws = {k.term for k in resume_report.keywords}
        jd_kws = {k.term for k in jd_report.keywords if k.tfidf_score > 0.001}
        matched_kws = sorted(resume_kws & jd_kws)
        missing_kws = sorted(jd_kws - resume_kws)

        return MatchReport(
            overall_match_pct=min(100.0, overall),
            dimensions=dimensions,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            bonus_skills=bonus_skills,
            matched_keywords=matched_kws[:30],
            missing_keywords=missing_kws[:30],
            matched_certs=matched_certs,
            missing_certs=missing_certs,
            jd_seniority=self._detect_seniority(jd_text),
            jd_required_degree=self._detect_required_degree(jd_text),
            keyword_details=kw_details,
        )

    def gap_summary(self, report: MatchReport) -> str:
        """Return a human-readable gap summary string."""
        lines = [
            f"Overall match: {report.overall_match_pct:.1f}%",
            f"Seniority required: {report.jd_seniority}",
            f"Degree required: {report.jd_required_degree}",
            "",
            "--- Skill gaps ---",
            f"  Matched  : {', '.join(report.matched_skills) or 'none'}",
            f"  Missing  : {', '.join(report.missing_skills) or 'none'}",
            f"  Bonus    : {', '.join(report.bonus_skills[:10]) or 'none'}",
            "",
            "--- Certification gaps ---",
            f"  Matched  : {', '.join(report.matched_certs) or 'none'}",
            f"  Missing  : {', '.join(report.missing_certs) or 'none'}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------
    def _score_skill_overlap(
        self, resume: KeywordReport, jd: KeywordReport
    ) -> DimensionScore:
        resume_skills = set(resume.top_technical)
        jd_skills = set(jd.top_technical)

        if not jd_skills:
            raw = 1.0
            details = {"note": "No technical skills detected in JD"}
        else:
            matched = resume_skills & jd_skills
            raw = len(matched) / len(jd_skills)
            details = { # type: ignore
                "jd_skills_count": len(jd_skills),
                "resume_skills_count": len(resume_skills),
                "matched_count": len(matched),
                "matched": sorted(matched),
                "missing": sorted(jd_skills - resume_skills),
            }

        weight = self.weights["skill_overlap"]
        return DimensionScore("skill_overlap", raw, weight, raw * weight, details)

    def _score_keyword_coverage(
        self,
        resume: KeywordReport,
        jd: KeywordReport,
        resume_text: str,
        jd_text: str,
    ) -> tuple[DimensionScore, list[KeywordMatchDetail]]:
        # Take top-scoring JD keywords as the target set
        jd_top = sorted(jd.keywords, key=lambda k: -k.tfidf_score)[:40]
        resume_terms = {k.term for k in resume.keywords}
        resume_text_lower = resume_text.lower()

        details_list: list[KeywordMatchDetail] = []
        hits = 0
        for kw in jd_top:
            # Direct term match OR substring match for compound phrases
            in_resume = kw.term in resume_terms or kw.term in resume_text_lower
            importance = (
                "required" if kw.tfidf_score > 0.005
                else "preferred" if kw.tfidf_score > 0.002
                else "bonus"
            )
            details_list.append(KeywordMatchDetail(kw.term, in_resume, True, importance))
            if in_resume:
                hits += 1

        raw = hits / len(jd_top) if jd_top else 1.0
        weight = self.weights["keyword_coverage"]
        return (
            DimensionScore(
                "keyword_coverage",
                raw,
                weight,
                raw * weight,
                {"jd_keywords_evaluated": len(jd_top), "matched": hits},
            ),
            details_list,
        )

    def _score_experience(self, resume_text: str, jd_text: str) -> DimensionScore:
        """
        Heuristic: extract years-of-experience mentions and compare.
        Also checks seniority signal alignment.
        """
        jd_years = self._extract_years_required(jd_text)
        resume_years = self._extract_years_in_resume(resume_text)
        jd_level = self._detect_seniority(jd_text)
        resume_level = self._detect_seniority(resume_text)

        if jd_years == 0:
            year_score = 1.0
        elif resume_years == 0:
            year_score = 0.5   # unknown; don't penalise too harshly
        else:
            year_score = min(1.0, resume_years / jd_years)

        # Small bonus for matching seniority level
        level_bonus = 0.1 if (jd_level != "unspecified" and jd_level == resume_level) else 0.0
        raw = min(1.0, year_score + level_bonus)

        weight = self.weights["experience_signals"]
        return DimensionScore(
            "experience_signals",
            raw,
            weight,
            raw * weight,
            {
                "jd_years_required": jd_years,
                "resume_years_detected": resume_years,
                "jd_seniority": jd_level,
                "resume_seniority_signals": resume_level,
            },
        )

    def _score_education(self, resume_text: str, jd_text: str) -> DimensionScore:
        jd_degree_level = self._degree_level(jd_text)
        resume_degree_level = self._degree_level(resume_text)

        if jd_degree_level == 0:
            raw = 1.0
            note = "No degree requirement in JD"
        elif resume_degree_level == 0:
            raw = 0.4
            note = "Could not detect degree in resume"
        elif resume_degree_level >= jd_degree_level:
            raw = 1.0
            note = "Degree requirement met or exceeded"
        else:
            raw = resume_degree_level / jd_degree_level
            note = "Degree level below JD requirement"

        # Field-of-study overlap (CS, engineering, business, etc.)
        field_score = self._field_overlap(resume_text, jd_text)
        raw = min(1.0, raw * 0.8 + field_score * 0.2)

        weight = self.weights["education_match"]
        return DimensionScore(
            "education_match",
            raw,
            weight,
            raw * weight,
            {
                "jd_degree_level": jd_degree_level,
                "resume_degree_level": resume_degree_level,
                "note": note,
                "field_overlap_score": round(field_score, 3),
            },
        )

    def _score_certifications(
        self, resume_text: str, jd_text: str
    ) -> tuple[DimensionScore, list[str], list[str]]:
        jd_certs = self.analyzer.extract_certifications(jd_text)
        resume_certs = self.analyzer.extract_certifications(resume_text)
        resume_cert_set = set(resume_certs)

        matched = [c for c in jd_certs if c in resume_cert_set]
        missing = [c for c in jd_certs if c not in resume_cert_set]

        if not jd_certs:
            raw = 1.0
        else:
            raw = len(matched) / len(jd_certs)

        weight = self.weights["certification_match"]
        return (
            DimensionScore(
                "certification_match",
                raw,
                weight,
                raw * weight,
                {"jd_certs": jd_certs, "matched": matched, "missing": missing},
            ),
            matched,
            missing,
        )

    # ------------------------------------------------------------------
    # Helper extractors
    # ------------------------------------------------------------------
    def _extract_years_required(self, text: str) -> int:
        """Parse '5+ years', '3-5 years', 'at least 4 years' patterns from JD."""
        text_lower = text.lower()
        patterns = [
            r'(\d+)\+\s*years?',
            r'(\d+)\s*[-–]\s*\d+\s*years?',
            r'at least\s+(\d+)\s*years?',
            r'minimum\s+(?:of\s+)?(\d+)\s*years?',
            r'(\d+)\s*years?\s+(?:of\s+)?experience',
        ]
        years: list[int] = []
        for pat in patterns:
            for m in re.finditer(pat, text_lower):
                try:
                    years.append(int(m.group(1)))
                except ValueError:
                    pass
        return max(years) if years else 0

    def _extract_years_in_resume(self, text: str) -> int:
        """Estimate years of experience from resume date ranges."""
        # Look for year spans like 2018 – 2023 or 2018-Present
        year_pattern = re.compile(
            r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|present|current|now)',
            re.IGNORECASE,
        )
        import datetime
        current_year = datetime.date.today().year
        total = 0
        for m in year_pattern.finditer(text):
            start = int(m.group(1))
            end_raw = m.group(2).lower()
            end = current_year if end_raw in ("present", "current", "now") else int(end_raw)
            total += max(0, end - start)
        # Cap at a sensible max to avoid inflated numbers
        return min(total, 35)

    def _detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        for level, signals in SENIORITY_PATTERNS.items():
            if any(s in text_lower for s in signals):
                return level
        return "unspecified"

    def _degree_level(self, text: str) -> int:
        text_lower = text.lower()
        max_level = 0
        for degree, level in DEGREE_SIGNALS.items():
            if degree in text_lower:
                max_level = max(max_level, level)
        return max_level

    def _detect_required_degree(self, jd_text: str) -> str:
        text_lower = jd_text.lower()
        for degree in sorted(DEGREE_SIGNALS, key=lambda d: -DEGREE_SIGNALS[d]):
            if degree in text_lower:
                return degree
        return "unspecified"

    def _field_overlap(self, resume_text: str, jd_text: str) -> float:
        fields = [
            "computer science", "software engineering", "information technology",
            "data science", "machine learning", "electrical engineering",
            "mathematics", "statistics", "business administration",
            "finance", "economics", "physics", "cybersecurity",
        ]
        r = resume_text.lower()
        j = jd_text.lower()
        jd_fields = {f for f in fields if f in j}
        if not jd_fields:
            return 1.0
        resume_fields = {f for f in fields if f in r}
        return len(resume_fields & jd_fields) / len(jd_fields)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Dimension weights must sum to 1.0, got {total:.3f}. "
                f"Adjust DIMENSION_WEIGHTS in jd_matcher.py."
            )
