"""
keyword_analyzer.py
====================
NLP module for extracting and analyzing keywords from resume and job-description text.

Responsibilities
----------------
- Extract technical skills, soft skills, certifications, and domain keywords
- Compute TF-IDF-weighted importance scores for each keyword
- Identify bigrams / trigrams (multi-word phrases) as compound keywords
- Return structured KeywordReport objects

Dependencies: re, collections, math  (pure stdlib — no external NLP libraries required)
Optional:     nltk stopwords (falls back to built-in list if unavailable)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Stopword list (used when nltk is not installed)
# ---------------------------------------------------------------------------
_BUILTIN_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "it",
    "its", "our", "your", "their", "my", "as", "if", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "than", "then", "when",
    "where", "who", "which", "what", "how", "all", "any", "each", "every",
    "more", "most", "other", "such", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "once", "also", "well", "just", "very", "too",
    "own", "same", "about", "up", "use", "used", "using", "get", "make",
    "work", "new", "good", "high", "large", "small", "long", "great",
    "strong", "able", "within", "across", "without", "including", "based",
    "related", "key", "various", "per", "etc", "eg", "ie",
}

# ---------------------------------------------------------------------------
# Curated skill dictionaries
# ---------------------------------------------------------------------------
TECH_SKILLS: set[str] = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "kotlin", "swift", "scala", "ruby", "php", "r", "matlab",
    "perl", "bash", "shell", "powershell", "dart", "elixir", "haskell",
    # Web / frontend
    "react", "angular", "vue", "svelte", "nextjs", "nuxtjs", "html", "css",
    "sass", "less", "webpack", "vite", "redux", "graphql", "rest", "soap",
    # Backend / frameworks
    "django", "flask", "fastapi", "spring", "nodejs", "express", "rails",
    "laravel", "asp.net", "gin", "fiber", "nestjs",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "oracle", "mssql", "neo4j", "influxdb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "github actions", "gitlab ci", "helm", "prometheus",
    "grafana", "datadog", "cloudformation", "pulumi",
    # Data / ML / AI
    "pandas", "numpy", "scipy", "sklearn", "scikit-learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "huggingface", "langchain",
    "spark", "hadoop", "airflow", "dbt", "kafka", "flink", "tableau",
    "power bi", "looker", "mlflow", "kubeflow", "opencv",
    # Tools / practices
    "git", "linux", "agile", "scrum", "kanban", "jira", "confluence",
    "microservices", "api", "oauth", "jwt", "grpc", "rabbitmq", "celery",
    "nginx", "apache", "linux", "unix", "vim",
}

CERTIFICATIONS: set[str] = {
    "aws certified", "azure certified", "gcp certified", "google certified",
    "comptia", "cissp", "cisa", "cism", "pmp", "prince2", "csm", "cka",
    "ckad", "cks", "rhcsa", "rhce", "lpic", "ccna", "ccnp", "ccie",
    "tensorflow developer", "professional data engineer", "solutions architect",
    "devops engineer", "cloud practitioner", "associate developer",
    "machine learning specialty", "security specialty", "database specialty",
}

SOFT_SKILLS: set[str] = {
    "leadership", "communication", "teamwork", "collaboration", "problem solving",
    "critical thinking", "analytical", "creativity", "adaptability", "time management",
    "project management", "mentoring", "coaching", "presentation", "negotiation",
    "stakeholder management", "cross-functional", "customer focus",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Keyword:
    """A single extracted keyword with metadata."""
    term: str
    frequency: int
    tfidf_score: float
    category: str          # "technical" | "certification" | "soft_skill" | "domain"
    is_ngram: bool = False


@dataclass
class KeywordReport:
    """Complete keyword analysis result."""
    source: str                                  # "resume" or "job_description"
    total_tokens: int
    unique_terms: int
    keywords: list[Keyword] = field(default_factory=list) # type: ignore
    top_technical: list[str] = field(default_factory=list) # type: ignore
    certifications_found: list[str] = field(default_factory=list) # type: ignore
    soft_skills_found: list[str] = field(default_factory=list) # type: ignore
    domain_keywords: list[str] = field(default_factory=list) # type: ignore

    def to_dict(self) -> dict: # type: ignore
        return {
            "source": self.source,
            "total_tokens": self.total_tokens,
            "unique_terms": self.unique_terms,
            "top_technical_skills": self.top_technical,
            "certifications": self.certifications_found,
            "soft_skills": self.soft_skills_found,
            "domain_keywords": self.domain_keywords,
            "all_keywords": [
                {
                    "term": k.term,
                    "frequency": k.frequency,
                    "tfidf_score": round(k.tfidf_score, 4),
                    "category": k.category,
                    "is_ngram": k.is_ngram,
                }
                for k in sorted(self.keywords, key=lambda x: -x.tfidf_score)
            ],
        } # type: ignore


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------
class KeywordAnalyzer:
    """
    Extracts and scores keywords from text using TF-IDF and curated skill lists.

    Usage
    -----
    >>> analyzer = KeywordAnalyzer()
    >>> report = analyzer.analyze(text, source="resume")
    >>> print(report.top_technical)
    """

    def __init__(self, custom_skills: Optional[set[str]] = None) -> None:
        self.tech_skills = TECH_SKILLS | (custom_skills or set())
        self.stopwords = self._load_stopwords()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, text: str, source: str = "resume") -> KeywordReport:
        """
        Full keyword analysis pipeline.

        Parameters
        ----------
        text   : Raw text extracted from resume or JD
        source : Label for the report ("resume" or "job_description")

        Returns
        -------
        KeywordReport with all extracted keywords, scores, and categories
        """
        text_clean = self._clean_text(text)
        tokens = self._tokenize(text_clean)
        unigrams = self._filter_tokens(tokens)
        bigrams = self._extract_ngrams(unigrams, n=2)
        trigrams = self._extract_ngrams(unigrams, n=3)

        # Frequency maps
        uni_freq = Counter(unigrams)
        bi_freq = Counter(bigrams)
        tri_freq = Counter(trigrams)

        # Combine all candidates
        all_terms: dict[str, tuple[int, bool]] = {}
        for term, freq in uni_freq.items():
            all_terms[term] = (freq, False)
        for term, freq in bi_freq.items():
            if freq >= 1:
                all_terms[term] = (freq, True)
        for term, freq in tri_freq.items():
            if freq >= 1:
                all_terms[term] = (freq, True)

        # TF-IDF (single-document: IDF is approximated from term distribution)
        tfidf_scores = self._compute_tfidf(all_terms, len(unigrams))

        # Categorise and build Keyword objects
        keywords: list[Keyword] = []
        certs: list[str] = []
        soft: list[str] = []
        technical: list[str] = []
        domain: list[str] = []

        for term, (freq, is_ngram) in all_terms.items():
            category = self._categorize(term)
            kw = Keyword(
                term=term,
                frequency=freq,
                tfidf_score=tfidf_scores.get(term, 0.0),
                category=category,
                is_ngram=is_ngram,
            )
            keywords.append(kw)

            if category == "certification":
                certs.append(term)
            elif category == "soft_skill":
                soft.append(term)
            elif category == "technical":
                technical.append(term)
            elif category == "domain":
                domain.append(term)

        # Sort each list by TF-IDF
        score_map = {k.term: k.tfidf_score for k in keywords}
        technical.sort(key=lambda t: -score_map.get(t, 0))
        domain.sort(key=lambda t: -score_map.get(t, 0))

        return KeywordReport(
            source=source,
            total_tokens=len(tokens),
            unique_terms=len(all_terms),
            keywords=keywords,
            top_technical=technical[:20],
            certifications_found=certs,
            soft_skills_found=soft,
            domain_keywords=domain[:15],
        )

    def extract_skills_only(self, text: str) -> list[str]:
        """Quick extraction of technical skills — no full report overhead."""
        text_lower = text.lower()
        found = []
        for skill in self.tech_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill) # type: ignore
        return sorted(found) # type: ignore

    def extract_certifications(self, text: str) -> list[str]:
        """Scan text for known certification keywords."""
        text_lower = text.lower()
        found = []
        for c in CERTIFICATIONS:
            pattern = r'\b' + re.escape(c) + r'\b'
            if re.search(pattern, text_lower):
                found.append(c) # type: ignore
        return sorted(found) # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_stopwords(self) -> set[str]:
        try:
            from nltk.corpus import stopwords  # type: ignore
            return set(stopwords.words("english")) | _BUILTIN_STOPWORDS # type: ignore
        except Exception:
            return _BUILTIN_STOPWORDS

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        # Preserve compound tech terms (e.g. c++, c#, .net, ci/cd)
        text = re.sub(r'[^\w\s\+\#\.\/\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        return [
            t for t in tokens
            if t not in self.stopwords
            and len(t) > 1
            and not t.isdigit()
        ]

    def _extract_ngrams(self, tokens: list[str], n: int) -> list[str]:
        return [
            " ".join(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        ]

    def _compute_tfidf(
        self,
        term_freq_map: dict[str, tuple[int, bool]],
        doc_length: int,
    ) -> dict[str, float]:
        """
        TF-IDF approximation for single-document analysis.
        IDF is estimated as log(doc_length / (freq + 1)) to reward rarer terms.
        """
        if doc_length == 0:
            return {}
        scores: dict[str, float] = {}
        for term, (freq, _) in term_freq_map.items():
            tf = freq / doc_length
            # Boost known skills; penalise very common generic words
            idf = math.log((doc_length + 1) / (freq + 1)) + 1.0
            boost = 2.0 if self._categorize(term) == "technical" else 1.0
            scores[term] = tf * idf * boost
        return scores

    def _categorize(self, term: str) -> str:
        if term in CERTIFICATIONS:
            return "certification"
        if term in self.tech_skills:
            return "technical"
        if term in SOFT_SKILLS:
            return "soft_skill"
        return "domain"
