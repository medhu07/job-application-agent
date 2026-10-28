from dotenv import load_dotenv

load_dotenv()
from src.application_writer import generate_application


# Use a small fictional test resume.
# Replace it with your actual resume text after the test works.
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
    ],
    "education": ["B.Tech student"],
    "experience": [],
    "projects": [
        "RAG chatbot using LangChain, embeddings and FAISS",
        "Movie recommendation system using Python and Streamlit",
    ],
    "certifications": [],
    "target_roles": ["AI Engineer Intern"],
    "years_of_experience": None,
}

job = {
    "title": "AI Engineer Intern",
    "hiring": {
        "display": "Example Company"
    },
    "location": {
        "city": "Bengaluru"
    },
    "work_mode": "hybrid",
    "job_type": "internship",
    "summary": (
        "Seeking an intern familiar with Python, "
        "LLM applications and retrieval augmented generation."
    ),
    "url": "https://example.com/job",
}

draft = generate_application(
    resume_text=resume_text,
    candidate_profile=candidate_profile,
    job=job,
)

print("\nFIT SUMMARY")
print(draft.fit_summary)

print("\nPOSITIONING STRATEGY")
print(draft.positioning_strategy)

print("\nEMAIL SUBJECT")
print(draft.email_subject)

print("\nEMAIL BODY")
print(draft.email_body)

print("\nCOVER LETTER")
print(draft.cover_letter)

print("\nEVIDENCE")
for item in draft.key_evidence:
    print("•", item)

print("\nMISSING INFORMATION")
for item in draft.missing_information:
    print("•", item)

print("\nCLAIMS TO VERIFY")
for item in draft.claims_to_verify:
    print("•", item)
