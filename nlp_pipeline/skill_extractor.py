"""
skill_extractor.py
==================
Takes the sections dict from section_detector.py and extracts
structured skill information from it.

TWO APPROACHES COMBINED:
    1. Skills database matching  — check against a curated list of
                                   known technical skills/tools
    2. spaCy NER + POS tagging   — catch skills not in our database
                                   using linguistic patterns

WHY TWO APPROACHES?
    A skills database alone misses new frameworks (nobody updated
    the list with "LangChain" yet). NER alone has false positives
    ("Python" the country, "Java" the island). Together they
    complement each other perfectly.
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────
# SKILLS DATABASE
# In a real project this would live in data/skills_db.json
# and be loaded from file. We inline it here for simplicity.
#
# Structure:
#   "technical"  — programming languages, frameworks, libraries
#   "tools"      — software, platforms, DevOps tools
#   "soft"       — interpersonal / workplace skills
#   "domains"    — subject areas (ML, NLP, Cloud, etc.)
# ─────────────────────────────────────────────────────────────────────

SKILLS_DB = {
    "technical": [
        # Languages
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "ruby", "go", "rust", "swift", "kotlin", "scala", "r", "matlab",
        "php", "perl", "bash", "shell", "sql", "html", "css",
        # Web frameworks
        "django", "flask", "fastapi", "express", "react", "angular", "vue",
        "next.js", "nuxt", "spring", "spring boot", "laravel", "rails",
        "asp.net", "node.js", "nodejs",
        # Data / ML
        "numpy", "pandas", "matplotlib", "seaborn", "plotly",
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
        "spacy", "nltk", "hugging face", "transformers", "opencv",
        "xgboost", "lightgbm", "catboost",
        # Databases
        "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra",
        "elasticsearch", "dynamodb", "firebase", "supabase",
        # Cloud / DevOps
        "aws", "azure", "gcp", "google cloud", "heroku", "vercel",
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github actions", "gitlab ci", "ci/cd",
        # Other
        "rest api", "graphql", "grpc", "websocket", "microservices",
        "linux", "git", "github", "bitbucket",
    ],
    "tools": [
        "vs code", "visual studio", "pycharm", "intellij", "eclipse",
        "jupyter", "jupyter notebook", "google colab", "postman",
        "figma", "jira", "confluence", "notion", "slack", "trello",
        "excel", "power bi", "tableau", "looker",
        "nginx", "apache", "rabbitmq", "kafka", "celery",
    ],
    "soft": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "time management", "adaptability",
        "collaboration", "project management", "analytical",
        "attention to detail", "creative", "self-motivated",
        "fast learner", "team player", "interpersonal",
    ],
    "domains": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data science", "data analysis",
        "data engineering", "mlops", "devops", "cloud computing",
        "cybersecurity", "blockchain", "iot", "embedded systems",
        "full stack", "frontend", "backend", "mobile development",
        "android", "ios", "game development",
    ],
}

# Flatten all skills into a single lookup set for fast membership testing
# WHY A SET?
#   Checking "python" in a set is O(1).
#   Checking "python" in a list is O(n) — slow for 300+ skills.
_ALL_SKILLS_LOWER = {
    skill.lower(): (category, skill)
    for category, skills in SKILLS_DB.items()
    for skill in skills
}

# Build multi-word skill patterns sorted longest first
# WHY LONGEST FIRST?
#   "machine learning" must be matched before "machine" and "learning"
#   are matched separately. Greedy longest-match prevents fragmentation.
_MULTI_WORD_SKILLS = sorted(
    [s for s in _ALL_SKILLS_LOWER if " " in s],
    key=len,
    reverse=True
)


# ─────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────

def extract_skills(sections: dict) -> dict:
    """
    Takes the sections dict from section_detector.detect_sections()
    and returns structured skill data.

    Returns:
    {
        "technical": ["Python", "Django", "FastAPI", "PostgreSQL"],
        "tools":     ["Git", "Docker", "VS Code"],
        "soft":      ["communication", "teamwork"],
        "domains":   ["machine learning", "NLP"],
        "all":       ["Python", "Django", ...]   # combined unique list
    }
    """
    # We look at skills section first (highest confidence),
    # then scan experience and projects for additional skills
    # mentioned in context.
    priority_text = sections.get("skills", "")
    context_text = " ".join([
        sections.get("experience", ""),
        sections.get("projects", ""),
        sections.get("summary", ""),
    ])

    # Extract from skills section with high confidence
    found_priority = _extract_from_text(priority_text)

    # Extract from context sections (catches skills not in skills section)
    found_context = _extract_from_text(context_text)

    # Merge: priority skills + any additional from context
    merged = _merge_results(found_priority, found_context)

    # Deduplicate and clean
    result = {
        cat: sorted(set(skills))
        for cat, skills in merged.items()
    }
    result["all"] = sorted(set(
        skill
        for skills in result.values()
        for skill in skills
    ))

    return result


# ─────────────────────────────────────────────────────────────────────
# EXTRACTION ENGINE
# ─────────────────────────────────────────────────────────────────────

def _extract_from_text(text: str) -> dict:
    """
    Runs both extraction strategies on a piece of text and
    combines their results.
    """
    if not text.strip():
        return {"technical": [], "tools": [], "soft": [], "domains": []}

    db_results = _database_match(text)
    ner_results = _spacy_ner_extract(text)

    return _merge_results(db_results, ner_results)


def _database_match(text: str) -> dict:
    """
    Strategy 1: Match text against our skills database.

    APPROACH:
        - First, replace multi-word skills with a placeholder so
          "machine learning" is treated as one unit, not two words.
        - Then tokenize and check each token against the single-word
          skills set.

    WHY NORMALIZE?
        Resumes write skills inconsistently:
        "Python3", "python", "PYTHON", "Python 3" → all mean Python.
        We lowercase and strip version numbers before matching.
    """
    found = {"technical": [], "tools": [], "soft": [], "domains": []}
    text_lower = text.lower()

    # Pass 1: Match multi-word skills first (greedy longest match)
    matched_spans = []
    for skill in _MULTI_WORD_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        for match in re.finditer(pattern, text_lower):
            # Check this span doesn't overlap with an already-matched span
            start, end = match.start(), match.end()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                category, canonical = _ALL_SKILLS_LOWER[skill]
                found[category].append(canonical.title())

    # Pass 2: Single-word skills
    # Tokenize by splitting on non-alphanumeric characters
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b', text_lower)
    for token in tokens:
        # Normalize: strip trailing version numbers (python3 → python)
        normalized = re.sub(r'\d+$', '', token).strip('.')
        if normalized in _ALL_SKILLS_LOWER:
            category, canonical = _ALL_SKILLS_LOWER[normalized]
            found[category].append(canonical.title())

    return found


def _spacy_ner_extract(text: str) -> dict:
    """
    Strategy 2: Use spaCy to find skills not in our database.

    WHAT spaCy GIVES US:
        - doc.ents  → Named entities (ORG, PRODUCT, LANGUAGE, etc.)
        - token.pos_ → Part of speech (NOUN, PROPN, VERB, etc.)

    HOW WE USE IT:
        - PRODUCT entities → likely a tool or framework
          e.g. spaCy labels "TensorFlow", "Kubernetes" as PRODUCT
        - ORG entities in a skills context → company-named tools
          e.g. "AWS", "Google Cloud" often tagged as ORG
        - PROPN (proper noun) tokens in skills section → candidate skills

    WHY NOT RELY ON NER ALONE?
        spaCy's en_core_web_sm is trained on news/web text, not resumes.
        It will tag "Python" as a language (good!) but also sometimes
        tag "Communication" as an ORG (bad). Database matching anchors
        us; NER extends us.
    """
    found = {"technical": [], "tools": [], "soft": [], "domains": []}

    doc = nlp(text[:10000])  # Limit to 10k chars to keep it fast

    # Extract named entities that are likely skills/tools
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_lower = ent_text.lower()

        # Skip if already in our database (avoid duplicates)
        if ent_lower in _ALL_SKILLS_LOWER:
            continue

        # Skip very short or very long entities
        if len(ent_text) < 2 or len(ent_text) > 30:
            continue

        # PRODUCT → likely a software tool or framework
        if ent.label_ == "PRODUCT":
            found["tools"].append(ent_text)

        # LANGUAGE → programming language not in our DB
        elif ent.label_ == "LANGUAGE":
            found["technical"].append(ent_text)

    return found


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _merge_results(a: dict, b: dict) -> dict:
    """Merges two skill dicts, combining lists for each category."""
    merged = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        merged[key] = list(a.get(key, [])) + list(b.get(key, []))
    return merged


# ─────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate what section_detector would give us
    mock_sections = {
        "skills": """
            Python, Django, FastAPI, PostgreSQL, MySQL
            Git, Docker, Linux, VS Code, Postman
            Machine Learning, NLP, scikit-learn, TensorFlow
            React, JavaScript, HTML, CSS
        """,
        "experience": """
            Built REST APIs using FastAPI and deployed on AWS EC2.
            Used Docker and Kubernetes for containerization.
            Wrote unit tests with pytest and set up GitHub Actions CI/CD.
        """,
        "projects": """
            Resume Analyzer: NLP pipeline using spaCy and sentence-transformers.
            Trained a PyTorch model for text classification.
        """,
        "summary": """
            Final year CS student skilled in Python and full stack development.
            Experience with cloud computing on GCP and Azure.
        """
    }

    result = extract_skills(mock_sections)

    print("=" * 50)
    print(f"TECHNICAL  ({len(result['technical'])}): {result['technical']}")
    print(f"TOOLS      ({len(result['tools'])}):     {result['tools']}")
    print(f"SOFT       ({len(result['soft'])}):      {result['soft']}")
    print(f"DOMAINS    ({len(result['domains'])}):   {result['domains']}")
    print(f"\nTOTAL UNIQUE SKILLS: {len(result['all'])}")
    print("=" * 50)