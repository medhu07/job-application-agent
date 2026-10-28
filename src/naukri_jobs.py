
import os
import streamlit as st
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

ACTOR_ID = "Kv60SmyhkBOf9Yae7"


def search_naukri_jobs(
    query="",
    location="India",
    job_type="Any",
    posted_within_days=7,
    max_jobs=50,
):
    """Search Naukri through Apify and normalize the returned jobs."""

    token = os.getenv("APIFY_API_TOKEN")

    if not token:
        raise ValueError(
            "APIFY_API_TOKEN is missing. Check your project .env file."
        )

    keywords = [
        keyword.strip()
        for keyword in (query or "").split(",")
        if keyword.strip()
    ]

    if not keywords:
        keywords = [
            "Python Developer",
            "AI Engineer",
            "GenAI Intern",
        ]

    if job_type == "Internship":
        actor_job_type = "internship"
    elif job_type in ("Full time", "Any"):
        actor_job_type = "permanent"
    elif job_type == "Part time":
        actor_job_type = "part-time"
    elif job_type == "Contract":
        actor_job_type = "contract"
    else:
        actor_job_type = "permanent"

    run_input = {
        "includeJobDescription": True,
        "jobType": actor_job_type,
        "location": (location or "").strip() or "India",
        "maxJobs": min(max(int(max_jobs), 1), 100),
        "monitorMode": False,
        "postedWithinDays": str(posted_within_days),
        "searchKeywords": keywords,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["BUYPROXIES94952"],
        },
    }

    try:
        client = ApifyClient(token)

        run = client.actor(ACTOR_ID).call(
            run_input=run_input
        )

        if not run:
            raise RuntimeError(
                "The Apify Actor did not return a run. "
                "Check the Actor status in Apify Console."
            )

        # Support dictionary and model based Run objects.
        if isinstance(run, dict):
            run_data = run
        elif hasattr(run, "model_dump"):
            run_data = run.model_dump(by_alias=True)
        elif hasattr(run, "dict"):
            run_data = run.dict(by_alias=True)
        else:
            run_data = dict(run)

        dataset_id = (
            run_data.get("defaultDatasetId")
            or run_data.get("default_dataset_id")
        )

        if not dataset_id:
            raise RuntimeError(
                "The Apify run returned no dataset ID. "
                "Check the run details in Apify Console."
            )

        raw_jobs = list(
            client.dataset(dataset_id).iterate_items()
        )

    except Exception as error:
        raise RuntimeError(
            f"Apify request failed: {error}"
        ) from error

    jobs = []

    for item in raw_jobs:
        title = (
            item.get("title")
            or item.get("job_title")
            or item.get("jobTitle")
        )

        if not title:
            continue

        description = (
            item.get("jobDescription")
            or item.get("job_description")
            or item.get("description")
            or item.get("summary")
            or ""
        )

        apply_url = (
            item.get("apply_url")
            or item.get("applyUrl")
            or item.get("url")
            or item.get("job_url")
            or item.get("jobUrl")
            or ""
        )

        jobs.append({
            **item,
            "title": title,
            "company": (
                item.get("company")
                or item.get("companyName")
                or "Company not specified"
            ),
            "location": (
                item.get("location")
                or item.get("jobLocation")
                or "Location not specified"
            ),
            "experience_min_years": item.get(
                "experience_min_years",
                item.get("experienceMinYears"),
            ),
            "experience_max_years": item.get(
                "experience_max_years",
                item.get("experienceMaxYears"),
            ),
            "salary_text": (
                item.get("salary_text")
                or item.get("salary")
                or "Not disclosed"
            ),
            "work_mode": (
                item.get("work_mode")
                or item.get("workMode")
                or "Not specified"
            ),
            "job_type": (
                item.get("job_type")
                or item.get("jobType")
                or actor_job_type
            ),
            "description": description,
            "apply_url": apply_url,
            "url": apply_url,
            "source": "Naukri",
        })

    return jobs
