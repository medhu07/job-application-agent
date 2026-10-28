
from dotenv import load_dotenv

load_dotenv()

from src.application_writer import ApplicationDraft
from src.claim_validator import ClaimValidator


resume_text = """
B.Tech student.
Built a RAG chatbot using LangChain, embeddings and FAISS.
Created a movie recommendation system using Python and Streamlit.
Worked with Python, SQL and generative AI tools.
"""

candidate_profile = {
    "skills": [
        "Python",
        "LangChain",
        "FAISS",
        "Streamlit",
        "SQL",
        "Generative AI",
    ],
    "education": ["B.Tech student"],
    "experience": [],
    "projects": [
        "RAG chatbot using LangChain, embeddings and FAISS",
        "Movie recommendation system using Python and Streamlit",
    ],
    "target_roles": ["AI Engineer Intern"],
    "years_of_experience": None,
}

draft = ApplicationDraft(
    fit_summary=(
        "The candidate has relevant personal project experience "
        "with Python and retrieval augmented generation."
    ),
    key_evidence=[
        "Built a RAG chatbot using LangChain, embeddings and FAISS.",
        "Created a movie recommendation system using Python and Streamlit.",
    ],
    positioning_strategy=(
        "Emphasize relevant projects and transferable technical skills."
    ),
    email_subject="Application for AI Engineer Intern",
    email_body=(
        "I built a RAG chatbot using LangChain and FAISS. "
        "I also fine tuned retrieval pipelines and developed a "
        "collaborative filtering recommendation engine. "
        "I am proficient in Python and SQL."
    ),
    cover_letter=(
        "As a B.Tech student, I have worked on a RAG chatbot and "
        "a movie recommendation system. I used collaborative filtering "
        "and SQL for data management in my recommendation project."
    ),
    missing_information=[],
    claims_to_verify=[],
)

validator = ClaimValidator()
report = validator.validate(
    resume_text=resume_text,
    candidate_profile=candidate_profile,
    draft=draft,
)

print("\nCLAIM VALIDATION REPORT")
print("=" * 40)
print("Safe to use:", report.is_safe)

print("\nUnsupported claims:")
for claim in report.unsupported_claims:
    print("•", claim)

print("\nRisky phrases:")
for phrase in report.risky_phrases:
    print("•", phrase)

print("\nVerified facts:")
for fact in report.verified_facts:
    print("•", fact)

print("\nRecommended changes:")
for change in report.recommended_changes:
    print("•", change)
