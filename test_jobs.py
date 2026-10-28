
from src.job_search import search_jobs

try:
    jobs = search_jobs(
        query="data analyst",
        limit=10,
    )

    print("Jobs returned:", len(jobs))

    for index, job in enumerate(jobs[:5], start=1):
        print(f"\nJob {index}")
        print("Title:", job.get("title"))
        print("ID:", job.get("id"))
        print("Job type:", job.get("job_type"))

    if not jobs:
        print("The API returned zero listings for this query.")

except Exception as error:
    print("Search failed:", repr(error))
