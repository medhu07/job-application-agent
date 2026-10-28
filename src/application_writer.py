
import json
import os
import re
from collections import Counter

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 45.0
MAX_RESUME_SENTENCES = 16
MAX_RESUME_KEYWORDS = 50

# Remove common filler words only from the compact context sent to the LLM.
# The original resume is never modified or displayed in this form.
STOP_WORDS = set("""
a an the and or but if then else when while where which who whom whose
what whatever this that these those am is are was were be been being do does
did doing have has had having can could may might must shall should will would
of at by for with about against between into through during before after above
below to from up down in out on off over under again further once here there
all any both each few more most other some such only own same so than
too very s t just don now i me my myself we our ours ourselves you your yours
yourself yourselves he him his himself she her hers herself it its itself they
them their theirs themselves as until because while also per via within across
""".split())


class ApplicationDraft(BaseModel):
    fit_summary: str
    key_evidence: list[str] = Field(default_factory=list)
    positioning_strategy: str
    email_subject: str
    email_body: str
    missing_information: list[str] = Field(default_factory=list)
    claims_to_verify: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """
You are an honest, thoughtful and precise AI job application writer.

Your task is to create a personalized job application using BOTH:
1. The candidate's actual resume and structured profile.
2. The selected job's specific description and requirements.

STRICT TRUTHFULNESS RULES:

1. Only use facts supported by the resume or explicitly confirmed
   candidate information.

2. Never invent or exaggerate skills, employment, internships,
   projects, qualifications, achievements, metrics, certifications,
   responsibilities or years of experience.

3. Never describe an academic or personal project as professional
   employment.

4. Do not assume the candidate used a technology simply because
   the job description requests it.

5. Building a RAG chatbot does not automatically prove that the
   candidate fine tuned models, evaluated retrieval pipelines,
   measured accuracy or deployed a production system.

6. Never invent company research, recruiter names, referrals,
   professional relationships or company motivations.

7. Treat the job description as evidence of employer requirements,
   never as evidence of candidate qualifications.

8. If information is missing, do not guess. Add material gaps to
   missing_information or claims_to_verify.

9. Explain how genuine skills and projects relate to job requirements
   without claiming unproven professional experience.

10. Never invent numerical results or business impact.

11. Do not call the candidate an expert or proficient unless supported.

12. Use natural, concise and professional language. Avoid generic
    praise, exaggerated claims and repetitive sentences.

13. Do not flag facts as unverified if they are clearly supported
    by the supplied resume or candidate profile.

14. Use the personal_details object as the authoritative source for
    contact information. Whenever a name, phone number, email address,
    GitHub URL or LinkedIn URL is supplied, use that exact value in the
    application and never use a generic placeholder for that field.
    Use a placeholder only for a field that was not supplied. Never
    invent a candidate's personal details.

15. Do not invent hiring contacts or imply that a recipient is a
    particular recruiter unless verified information is provided.

16. For students and freshers, emphasize genuine projects, relevant
    skills, learning and transferable abilities.

APPLICATION EMAIL RULES:

1. Write an email specifically for the selected job and company.

2. Use the job title and company name in the subject when available.

3. Start with a professional greeting. Use a recruiter's name only
   when that name has been explicitly provided and verified.
   Otherwise use "Dear Hiring Team,".

4. In the opening, clearly state the role being applied for.

5. Select one or two of the candidate's strongest relevant examples
   and connect them directly to the job requirements.

6. Explain what the candidate can contribute without exaggeration.

7. Keep the email under 180 words.

8. Mention the attached resume, but do not claim that an attachment
   has already been added or sent.

9. End with a professional sign off. Include every supplied personal
   detail that is relevant to an email signature, using the exact values
   from personal_details. This can include name, phone, email, GitHub
   URL and LinkedIn URL.

10. Never output generic placeholders for fields that were supplied.
    If a field is missing, omit it or use a clear editable placeholder.

11. Keep the email body readable as plain text, suitable for Gmail.

OUTPUT REQUIREMENTS:

Return exactly one JSON object with these seven fields:

fit_summary: string
key_evidence: list of strings
positioning_strategy: string
email_subject: string
email_body: string
missing_information: list of strings
claims_to_verify: list of strings

Use empty arrays when no information is available.
Return valid JSON only, without Markdown fences or commentary.
"""


def prepare_job_details(job: dict) -> dict:
    """Extract available job details without inventing fields."""

    hiring = job.get("hiring") or {}
    location = job.get("location") or {}
    experience = job.get("experience") or {}
    salary = job.get("salary") or {}

    if isinstance(location, dict):
        location_value = (
            location.get("city")
            or location.get("display")
            or job.get("city")
            or "Not provided"
        )
    else:
        location_value = location or job.get("city") or "Not provided"

    if isinstance(experience, dict):
        experience_value = experience.get("label", "Not provided")
    else:
        experience_value = experience or "Not provided"

    if isinstance(salary, dict):
        salary_value = salary.get("label", "Not disclosed")
    else:
        salary_value = salary or "Not disclosed"

    company = (
        hiring.get("display")
        or hiring.get("employer")
        or job.get("company")
        or "Not provided"
    )

    description = (
        job.get("description")
        or job.get("job_description")
        or job.get("summary")
        or "No description provided"
    )

    return {
        "title": job.get("title", "Not provided"),
        "company": company,
        "description": description,
        "responsibilities": job.get("responsibilities", []),
        "requirements": job.get("requirements", []),
        "skills": job.get("skills", []),
        "location": location_value,
        "work_mode": job.get("work_mode", "Not provided"),
        "job_type": job.get("job_type", "Not provided"),
        "experience": experience_value,
        "salary": salary_value,
        "job_url": job.get("url", ""),
        "apply_url": job.get("apply_url", ""),
    }



def _words(text: str) -> list[str]:
    """Return meaningful words while retaining technical terms and numbers."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]*|\d+(?:\.\d+)?%?", text.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def compact_text(text: str, max_chars: int = 7000) -> str:
    """Remove common stop words and repeated whitespace from long text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]*|\d+(?:\.\d+)?%?|[^\w\s]", text)
    compacted = [token for token in tokens if token.lower() not in STOP_WORDS]
    result = " ".join(compacted)
    return result[:max_chars]


def build_resume_context(resume_text: str, job_details: dict, max_chars: int = 5000) -> dict:
    """Prioritize resume sentences relevant to the job and provide key terms."""
    text = re.sub(r"\s+", " ", resume_text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\s*[|•]\s+", text) if part.strip()]
    job_text = " ".join(str(job_details.get(key, "")) for key in ("title", "description", "responsibilities", "requirements", "skills"))
    job_terms = set(_words(job_text))

    scored = []
    for index, sentence in enumerate(sentences):
        terms = _words(sentence)
        overlap = len(set(terms) & job_terms)
        # Slightly favor useful resume lines with concrete evidence.
        evidence_bonus = sum(1 for pattern in (r"\d", r"project", r"built", r"develop", r"intern", r"python", r"sql", r"llm", r"machine learning") if re.search(pattern, sentence, re.I))
        scored.append((overlap * 2 + evidence_bonus, index, sentence))

    selected_indices = {index for _, index, _ in sorted(scored, reverse=True)[:MAX_RESUME_SENTENCES]}
    selected_sentences = [sentence for index, sentence in enumerate(sentences) if index in selected_indices]
    compact_resume = compact_text(". ".join(selected_sentences), max_chars=max_chars)

    counts = Counter(_words(text))
    job_relevant = [term for term, _ in counts.most_common() if term in job_terms]
    general_important = [term for term, _ in counts.most_common() if term not in job_terms]
    keywords = list(dict.fromkeys(job_relevant + general_important))[:MAX_RESUME_KEYWORDS]

    return {
        "resume_evidence_compact": compact_resume,
        "resume_keywords": keywords,
        "note": "Resume text is compacted to reduce token use. Treat keywords as search aids only; use resume evidence for factual claims.",
    }


def compact_job_details(job_details: dict) -> dict:
    """Keep useful job facts while compacting verbose descriptions."""
    compacted = dict(job_details)
    for key in ("description", "responsibilities", "requirements", "skills"):
        value = compacted.get(key)
        if isinstance(value, str):
            compacted[key] = compact_text(value, max_chars=1800)
        elif isinstance(value, list):
            compacted[key] = [compact_text(str(item), max_chars=180) for item in value if str(item).strip()][:12]
    return compacted


def extract_json_object(content: str) -> dict:
    """Parse a JSON object, tolerating surrounding Markdown fences."""

    if not content or not content.strip():
        raise ValueError("The AI returned an empty response.")

    content = content.strip()
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The AI did not return a complete JSON object.")

    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"The AI returned malformed JSON: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError("The AI response must be a JSON object.")

    return data


def apply_personal_details_to_draft(draft: ApplicationDraft, details: dict) -> ApplicationDraft:
    """Ensure supplied contact details replace generic AI placeholders."""
    replacements = {
        "{Your Name}": details.get("name"),
        "{Your Phone Number}": details.get("phone"),
        "{Your Email}": details.get("email"),
        "you@example.com": details.get("email"),
        "https://github.com/{username}": details.get("github_url"),
        "https://linkedin.com/in/{username}": details.get("linkedin_url"),
        "https://linkedin.com/in/username": details.get("linkedin_url"),
        "https://github.com/username": details.get("github_url"),
    }

    email_body = draft.email_body
    for placeholder, value in replacements.items():
        if value:
            email_body = email_body.replace(placeholder, value)

    # Ensure supplied contact details appear in the email signature even if
    # the model forgot to include them. Avoid adding duplicate lines.
    signature_lines = []
    for key in ("name", "phone", "email", "github_url", "linkedin_url"):
        value = details.get(key)
        if not value:
            continue
        if key == "github_url":
            line = f"GitHub: {value}"
        elif key == "linkedin_url":
            line = f"LinkedIn: {value}"
        else:
            line = value
        if value not in email_body:
            signature_lines.append(line)

    if signature_lines:
        email_body = email_body.rstrip() + "\n" + "\n".join(signature_lines)

    return draft.model_copy(update={"email_body": email_body})


def generate_application(
    resume_text: str,
    candidate_profile: dict,
    job: dict,
    personal_details: dict | None = None,
) -> ApplicationDraft:
    """
    Generate a personalized application draft.

    Creates drafts only. Never sends emails.
    personal_details is optional and can contain user supplied
    name, phone, email, GitHub URL and LinkedIn URL.
    """

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    if not isinstance(candidate_profile, dict):
        raise ValueError("Candidate profile must be a dictionary.")

    if not isinstance(job, dict) or not job.get("title"):
        raise ValueError("Valid job details are required.")

    personal_details = personal_details or {}

    allowed_personal_fields = (
        "name",
        "phone",
        "email",
        "github_url",
        "linkedin_url",
    )

    safe_personal_details = {
        key: str(personal_details[key]).strip()
        for key in allowed_personal_fields
        if personal_details.get(key)
    }

    job_details = compact_job_details(prepare_job_details(job))
    resume_context = build_resume_context(resume_text, job_details)

    user_payload = {
        "candidate_resume_context": resume_context,
        "structured_candidate_profile": candidate_profile,
        "job_details": job_details,
        "personal_details": safe_personal_details,
        "instructions": (
            "Generate an application for this exact job. "
            "Use the job description as the primary source for "
            "the employer's requirements. Select the most relevant "
            "evidence from the resume. Make the email concise and "
            "specific to the role. Use editable personal placeholders "
            "for any missing contact details. Do not invent facts. "
            "If the job description is unavailable, disclose that "
            "limitation in missing_information and avoid claiming "
            "that the application is tailored to unseen requirements. "
            "Return the required JSON object only."
        ),
    }

    # Generate one cold email draft with Gemini structured JSON output.
    # The API key is read from GEMINI_API_KEY in the environment or .env file.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=json.dumps(user_payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ApplicationDraft,
                max_output_tokens=1400,
                temperature=0,
            ),
        )

        content = response.text
        data = extract_json_object(content)
        draft = ApplicationDraft.model_validate(data)

        if not draft.email_subject.strip():
            raise ValueError("The generated email subject is empty.")

        if not draft.email_body.strip():
            raise ValueError("The generated email body is empty.")


        return apply_personal_details_to_draft(draft, safe_personal_details)

    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(
            f"Could not create a valid application draft: {error}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Application generation request failed: {error}"
        ) from error
