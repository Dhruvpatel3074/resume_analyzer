"""
demo_pipeline.py
================
End-to-end demo of the three NLP modules working together.

Run:
    python demo_pipeline.py                    # uses built-in sample texts
    python demo_pipeline.py resume.pdf jd.txt  # from files

Outputs:
    pipeline_result.json   — full structured JSON
    (stdout)               — human-readable summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8") # type: ignore

from resume_analyzer.nlp_pipeline.keyword_analyzer import KeywordAnalyzer
from resume_analyzer.nlp_pipeline.jd_matcher import JDMatcher
from resume_analyzer.nlp_pipeline.ats_scorer import ATSScorer, extract_text_from_file


# ---------------------------------------------------------------------------
# Sample texts (used when no files are provided)
# ---------------------------------------------------------------------------
SAMPLE_RESUME = """
Jane Doe
jane.doe@email.com | +1 (555) 123-4567 | linkedin.com/in/janedoe | github.com/janedoe

SUMMARY
Experienced Software Engineer with 5 years building scalable backend systems
and data pipelines using Python, AWS, and Kubernetes. AWS Certified Solutions Architect.

EXPERIENCE
Senior Software Engineer — TechCorp Inc.                       2021 – Present
• Designed and deployed microservices using Python (FastAPI) and Docker on AWS ECS
• Built real-time data pipelines with Apache Kafka and Apache Spark processing 2M events/day
• Reduced infrastructure costs 30% by migrating to Kubernetes with Terraform-managed infra
• Led a cross-functional team of 5 engineers; conducted code reviews and mentored juniors

Software Engineer — DataWave Ltd.                              2019 – 2021
• Developed REST APIs with Django and PostgreSQL serving 500k daily active users
• Implemented CI/CD pipelines using GitHub Actions and Jenkins
• Created ML feature pipelines using pandas, scikit-learn, and Airflow

EDUCATION
B.Sc. Computer Science — State University                      2015 – 2019
GPA: 3.8/4.0 | Relevant coursework: Algorithms, Distributed Systems, ML

SKILLS
Languages:   Python, Go, SQL, Bash, JavaScript
Frameworks:  FastAPI, Django, Flask, React
Cloud:       AWS (EC2, S3, Lambda, ECS, RDS), GCP basics
DevOps:      Docker, Kubernetes, Terraform, GitHub Actions, Jenkins, Helm
Databases:   PostgreSQL, MySQL, Redis, MongoDB, Elasticsearch
Data / ML:   Pandas, NumPy, Scikit-learn, Spark, Kafka, Airflow, dbt
Tools:       Git, Linux, Jira, Confluence, Prometheus, Grafana

CERTIFICATIONS
• AWS Certified Solutions Architect – Associate (2022)
• CKA: Certified Kubernetes Administrator (2023)

PROJECTS
Resume ATS Pipeline (2024)
• Built an NLP pipeline in Python to analyse resumes against job descriptions
• Used TF-IDF keyword extraction and cosine similarity for JD matching
"""

SAMPLE_JD = """
Senior Backend Engineer — Cloud Infrastructure

We are looking for a Senior Backend Engineer to join our platform team.

Requirements:
• 5+ years of professional software engineering experience
• Strong proficiency in Python or Go
• Experience with AWS (EC2, S3, Lambda, RDS) or GCP
• Kubernetes and Docker expertise — CKA certification is a plus
• Hands-on experience with Terraform or Pulumi for infrastructure as code
• Proficiency in PostgreSQL and Redis; MongoDB knowledge is a bonus
• Experience building real-time data pipelines (Kafka, Flink, or Spark)
• CI/CD experience with GitHub Actions or GitLab CI
• Strong understanding of microservices architecture and REST API design
• Experience with monitoring tools: Prometheus, Grafana, or Datadog
• Bachelor's degree in Computer Science, Software Engineering, or equivalent

Nice to have:
• Experience with Airflow or dbt for data orchestration
• Machine learning / MLOps exposure
• AWS Solutions Architect or DevOps Engineer certification

Soft skills:
• Leadership and mentoring ability
• Strong communication and collaboration skills
• Ability to work cross-functionally with product and data teams
"""


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(resume_text: str, jd_text: str) -> dict: # type: ignore
    """
    Execute all three modules and return a unified result dict.

    Steps
    -----
    1. KeywordAnalyzer  → extract skills / keywords from resume
    2. JDMatcher        → match resume against JD
    3. ATSScorer        → composite ATS score + suggestions
    """
    print("\n" + "─" * 60)
    print("  RESUME NLP PIPELINE")
    print("─" * 60)

    # ── Step 1: Keyword Analysis ──────────────────────────────────
    print("\n[1/3] Running keyword analysis…")
    analyzer = KeywordAnalyzer()
    resume_kw_report = analyzer.analyze(resume_text, source="resume")
    jd_kw_report = analyzer.analyze(jd_text, source="job_description")

    print(f"      Resume  → {len(resume_kw_report.top_technical)} tech skills, "
          f"{len(resume_kw_report.certifications_found)} certs detected")
    print(f"      JD      → {len(jd_kw_report.top_technical)} tech skills required")

    # ── Step 2: JD Matching ───────────────────────────────────────
    print("\n[2/3] Matching resume against JD…")
    matcher = JDMatcher()
    match_report = matcher.match(resume_text, jd_text)

    print(f"      Overall match  : {match_report.overall_match_pct:.1f}%")
    print(f"      Skills matched : {len(match_report.matched_skills)}")
    print(f"      Skills missing : {len(match_report.missing_skills)}")
    print(f"      JD seniority   : {match_report.jd_seniority}")
    print(f"      JD degree req  : {match_report.jd_required_degree}")

    # ── Step 3: ATS Scoring ───────────────────────────────────────
    print("\n[3/3] Computing ATS score…")
    scorer = ATSScorer()
    ats_result = scorer.score(resume_text, jd_text)

    # Print rich summary
    print()
    ats_result.print_summary()

    # ── Build unified output ──────────────────────────────────────
    result = { # type: ignore
        "pipeline_version": "1.0.0",
        "modules_used": ["keyword_analyzer", "jd_matcher", "ats_scorer"],
        "keyword_analysis": {
            "resume": resume_kw_report.to_dict(), # type: ignore
            "job_description": jd_kw_report.to_dict(), # type: ignore
        },
        "jd_match": match_report.to_dict(), # type: ignore
        "ats_result": ats_result.to_dict(), # type: ignore
    }
    return result # type: ignore


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = sys.argv[1:]

    if len(args) >= 2:
        # File mode
        resume_path, jd_path = args[0], args[1]
        print(f"Loading resume from: {resume_path}")
        print(f"Loading JD from    : {jd_path}")
        resume_text = extract_text_from_file(resume_path)
        jd_text = Path(jd_path).read_text(encoding="utf-8", errors="replace")
    else:
        # Demo mode
        print("No files provided — using built-in sample resume and JD.")
        resume_text = SAMPLE_RESUME
        jd_text = SAMPLE_JD

    result = run_pipeline(resume_text, jd_text) # type: ignore

    # Save JSON output
    out_path = Path("pipeline_result.json")
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✓ Full JSON saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
