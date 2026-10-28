
import json
import re
from groq import Groq

MODEL = "openai/gpt-oss-20b"


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(item) for item in value.values())
    return re.sub(r"\s+", " ", str(value)).strip()


def search_jobs(
    query="",
    location=None,
    work_mode=None,
    job_type=None,
    experience_level=None,
    posted_within_days=7,
    limit=20,
):
    from src.naukri_jobs import search_naukri_jobs

    return search_naukri_jobs(
        query=query or "",
        location=location or "India",
        job_type=job_type or "Any",
        posted_within_days=posted_within_days,
        max_jobs=limit,
    )


def get_experience_range(job):
    """Read numeric experience fields first, then try text fields."""

    minimum = job.get("experience_min_years")
    maximum = job.get("experience_max_years")

    try:
        minimum = float(minimum) if minimum is not None else None
    except (TypeError, ValueError):
        minimum = None

    try:
        maximum = float(maximum) if maximum is not None else None
    except (TypeError, ValueError):
        maximum = None

    if minimum is not None and maximum is not None:
        return minimum, maximum

    text = normalize_text(
        job.get("experience")
        or job.get("experience_text")
        or job.get("experience_required")
        or ""
    )

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:to|[-–])\s*"
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        text,
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1)), float(match.group(2))

    return minimum, maximum


def filter_jobs(
    jobs,
    min_experience=0,
    max_experience=None,
    job_type="Any",
):
    """Filter by experience overlap and employment type."""

    filtered = []

    min_experience = float(min_experience or 0)
    max_experience = (
        float(max_experience)
        if max_experience is not None
        else None
    )

    type_aliases = {
        "Internship": ["internship", "intern"],
        "Full time": ["permanent", "full time", "full-time", "fulltime"],
        "Part time": ["part time", "part-time", "parttime"],
        "Contract": ["contract", "contractor", "freelance"],
    }

    for original in jobs:
        job = dict(original)

        required_min, required_max = get_experience_range(job)

        # Missing experience data should not automatically exclude a job.
        if required_min is None or required_max is None:
            job["experience_match"] = "Not specified"
        else:
            # Keep jobs whose required experience range overlaps the
            # user's selected experience range.
            if (
                max_experience is not None
                and required_min > max_experience
            ):
                continue

            if required_max < min_experience:
                continue

            job["experience_match"] = "Potential match"

        # Any means do not filter by job type.
        if job_type != "Any":
            aliases = type_aliases.get(job_type, [])
            actual_type = normalize_text(
                job.get("job_type")
            ).lower()

            title = normalize_text(job.get("title")).lower()

            if actual_type:
                if not any(alias in actual_type for alias in aliases):
                    continue
            elif not any(alias in title for alias in aliases):
                # Preserve uncertain metadata instead of discarding it.
                job["job_type_match"] = "Needs review"
            else:
                job["job_type_match"] = "Likely match"

        filtered.append(job)

    return filtered


def fetch_full_job_description(job):
    """Use the description returned by Apify, without LocoJobs calls."""

    job = dict(job)

    description = normalize_text(
        job.get("description")
        or job.get("jobDescription")
        or job.get("job_description")
        or job.get("summary")
    )

    job["description"] = description

    job["jd_fetch_status"] = (
        "Description available"
        if len(description) >= 80
        else "Description incomplete"
    )

    return job


def parse_json_response(content):
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise ValueError("AI response did not contain valid JSON.")

    return json.loads(content[start:end + 1])


def analyze_job_fit(
    resume_text,
    candidate_profile,
    job,
    target_role=None,
):
    job_details = {
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "location": job.get("location", ""),
        "description": job.get("description", ""),
        "summary": job.get("summary", ""),
        "responsibilities": job.get("responsibilities", ""),
        "requirements": job.get(
            "requirements",
            job.get("qualifications", ""),
        ),
        "skills": job.get("skills", ""),
        "job_type": job.get("job_type", ""),
        "experience_min_years": job.get("experience_min_years"),
        "experience_max_years": job.get("experience_max_years"),
    }

    if len(normalize_text(job_details)) < 80:
        raise ValueError(
            "The available job description is too short "
            "for reliable comparison."
        )

    prompt = {
        "resume": resume_text,
        "candidate_profile": candidate_profile or {},
        "target_role_and_keywords": target_role or "",
        "job_description": job_details,
        "instructions": [
            "Compare job requirements with evidence in the resume.",
            "Do not judge fit using title or shared keywords alone.",
            "Never invent skills, experience, projects, or qualifications.",
            "Missing evidence is not proof that the candidate lacks a skill.",
            "Explain the most important matches and gaps.",
            "The score measures resume alignment, not hiring probability.",
            "Return valid JSON only.",
        ],
        "output_schema": {
            "fit_score": "integer from 0 to 100",
            "confidence": "Low, Medium, or High",
            "role_relevance": "Relevant, Related, or Unrelated",
            "summary": "Brief explanation",
            "matching_requirements": [
                {
                    "requirement": "Job requirement",
                    "resume_evidence": "Evidence in resume",
                }
            ],
            "missing_requirements": ["Unproven requirements"],
            "concerns": ["Important concerns"],
            "recommendation": "Apply, Consider, or Low priority",
        },
    }

    client = Groq()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous job fit analyst. "
                    "Use only supplied evidence. Return valid JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        max_completion_tokens=2200,
        temperature=0,
    )

    result = parse_json_response(
        response.choices[0].message.content
    )

    result["fit_score"] = max(
        0,
        min(100, int(result.get("fit_score", 0))),
    )

    if result.get("role_relevance") == "Unrelated":
        result["fit_score"] = min(result["fit_score"], 20)
        result["recommendation"] = "Low priority"

    if job.get("jd_fetch_status") != "Description available":
        result["confidence"] = "Low"
        result.setdefault("concerns", []).append(
            "The full job description could not be confirmed."
        )

    return result


def rank_jobs(
    jobs,
    candidate_profile=None,
    preferences=None,
    resume_text="",
):
    ranked = []

    for original in jobs:
        job = fetch_full_job_description(original)

        try:
            analysis = analyze_job_fit(
                resume_text=resume_text,
                candidate_profile=candidate_profile or {},
                job=job,
                target_role=(preferences or {}).get("target_role"),
            )

            job["fit_analysis"] = analysis
            job["relevance_score"] = analysis["fit_score"]
            job["analysis_status"] = "Completed"

        except Exception as error:
            job["fit_analysis"] = None
            job["relevance_score"] = None
            job["analysis_status"] = str(error)

        ranked.append(job)

    ranked.sort(
        key=lambda item: (
            item.get("relevance_score")
            if item.get("relevance_score") is not None
            else -1
        ),
        reverse=True,
    )

    return ranked
