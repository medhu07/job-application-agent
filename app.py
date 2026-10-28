
import os
from urllib.parse import urlencode

import requests
import streamlit as st
from dotenv import load_dotenv

from src.naukri_jobs import search_naukri_jobs
from src.resume_parser import extract_resume_text, parse_resume_with_llm
from src.job_search import filter_jobs, rank_jobs
from src.application_writer import generate_application
from src.application_guard import ApplicationGuard


load_dotenv()

st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Job Discovery and Application Agent")
st.write(
    "Find relevant jobs, compare them with your resume, "
    "and prepare personalized application drafts."
)


# Initialize session state
defaults = {
    "resume_text": None,
    "candidate_profile": None,
    "preferences": None,
    "jobs": [],
    "application_results": {},
    "resume_filename": None,
    "recruiter_contacts": {},
    "personal_details": {},
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# Personal details
st.header("Your application details")
st.caption(
    "Enter your details once. They will be reused when generating "
    "application drafts during this session."
)

saved_details = st.session_state["personal_details"]

with st.expander("Edit personal details", expanded=False):
    personal_name = st.text_input(
        "Your name",
        value=saved_details.get("name", ""),
        placeholder="Your Name",
        key="personal_name_input",
    )
    personal_phone = st.text_input(
        "Phone number",
        value=saved_details.get("phone", ""),
        placeholder="Your Phone Number",
        key="personal_phone_input",
    )
    personal_email = st.text_input(
        "Email address",
        value=saved_details.get("email", ""),
        placeholder="you@example.com",
        key="personal_email_input",
    )
    personal_github = st.text_input(
        "GitHub URL",
        value=saved_details.get("github_url", ""),
        placeholder="https://github.com/username",
        key="personal_github_input",
    )
    personal_linkedin = st.text_input(
        "LinkedIn URL",
        value=saved_details.get("linkedin_url", ""),
        placeholder="https://linkedin.com/in/username",
        key="personal_linkedin_input",
    )

    if st.button("Save personal details", key="save_personal_details"):
        st.session_state["personal_details"] = {
            "name": personal_name.strip(),
            "phone": personal_phone.strip(),
            "email": personal_email.strip(),
            "github_url": personal_github.strip(),
            "linkedin_url": personal_linkedin.strip(),
        }
        st.success("Personal details saved for this session.")


def find_recruiter_contacts(company_name):
    """Find publicly sourced HR or recruiting contacts through Hunter."""

    api_key = os.getenv("HUNTER_API_KEY", "").strip()

    if not api_key:
        return {
            "error": (
                "HUNTER_API_KEY was not found. "
                "Check your .env file."
            )
        }

    if not company_name or company_name.strip().lower() in {
        "company not specified",
        "unknown",
        "n/a",
        "none",
    }:
        return {
            "error": "This listing does not provide a usable company name."
        }

    base_url = "https://api.hunter.io/v2/domain-search"

    params = {
        "company": company_name.strip(),
        "department": "hr",
        "limit": 10,
        "api_key": api_key,
    }

    try:
        response = requests.get(
            base_url,
            params=params,
            timeout=20,
        )

        if response.status_code == 401:
            return {
                "error": (
                    "Hunter rejected the API key. "
                    "Check HUNTER_API_KEY in your .env file."
                )
            }

        if response.status_code == 429:
            return {
                "error": (
                    "Hunter API usage limit reached. "
                    "Check your plan and available credits."
                )
            }

        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", {})
        emails = data.get("emails", []) or []

        if not emails:
            fallback_params = {
                "company": company_name.strip(),
                "limit": 10,
                "api_key": api_key,
            }

            fallback = requests.get(
                base_url,
                params=fallback_params,
                timeout=20,
            )
            fallback.raise_for_status()

            if fallback.status_code == 401:
                return {"error": "Hunter rejected the API key."}

            if fallback.status_code == 429:
                return {"error": "Hunter API usage limit reached."}

            payload = fallback.json()
            data = payload.get("data", {})
            emails = data.get("emails", []) or []

        recruiter_terms = (
            "recruit",
            "talent acquisition",
            "human resources",
            "hr ",
            " hr",
            "hiring",
            "people partner",
            "staffing",
            "resourcing",
        )

        contacts = []

        for item in emails:
            position = str(item.get("position") or "").strip()
            department = str(item.get("department") or "").strip()
            searchable = f"{position} {department}".lower()

            if any(term in searchable for term in recruiter_terms):
                verification_data = item.get("verification") or {}

                email = item.get("value") or item.get("email") or ""

                if not email:
                    continue

                contacts.append({
                    "email": email,
                    "name": (
                        f"{item.get('first_name') or ''} "
                        f"{item.get('last_name') or ''}"
                    ).strip(),
                    "position": position or "HR or recruiting contact",
                    "verification": verification_data.get(
                        "status", "Not provided"
                    ),
                    "confidence": item.get("confidence"),
                    "sources": item.get("sources") or [],
                })

        contacts.sort(
            key=lambda contact: (
                str(contact["verification"]).lower() == "valid",
                contact["confidence"]
                if isinstance(contact["confidence"], (int, float))
                else 0,
            ),
            reverse=True,
        )

        return {
            "contacts": contacts[:5],
            "domain": data.get("domain") or "",
            "sources": data.get("websites") or [],
            "message": (
                "No HR or recruiting contact was found for this company."
                if not contacts else ""
            ),
        }

    except requests.RequestException as error:
        return {"error": f"Could not contact Hunter: {error}"}

    except (ValueError, TypeError, AttributeError) as error:
        return {"error": f"Hunter returned an unexpected response: {error}"}


def gmail_compose_url(recipient, subject, body):
    """Prepare a Gmail compose page for manual review and sending."""

    query = urlencode({
        "view": "cm",
        "fs": "1",
        "to": recipient,
        "su": subject,
        "body": body,
    })

    return f"https://mail.google.com/mail/?{query}"


# Section 1: Resume upload and analysis
st.header("1. Your resume")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    key="resume_uploader",
)

if st.session_state["candidate_profile"]:

    st.success(
        "Resume analyzed: "
        f"{st.session_state['resume_filename'] or 'Resume loaded'}"
    )

    st.caption(
        "Your profile is saved for this session. "
        "You can search for jobs without analyzing the resume again."
    )

    with st.expander("View your analyzed profile"):
        st.json(st.session_state["candidate_profile"])

    if st.button("Analyze a different resume"):
        st.session_state["candidate_profile"] = None
        st.session_state["resume_text"] = None
        st.session_state["resume_filename"] = None
        st.session_state["jobs"] = []
        st.session_state["application_results"] = {}
        st.session_state["recruiter_contacts"] = {}
        st.rerun()

else:

    if st.button("Analyze resume", type="primary"):

        if uploaded_file is None:
            st.error("Please upload your resume first.")

        else:
            try:
                with st.spinner("Analyzing your resume..."):
                    resume_text = extract_resume_text(uploaded_file)

                    if not resume_text.strip():
                        st.error(
                            "No readable text was found. "
                            "Try a text based PDF or DOCX file."
                        )
                        st.stop()

                    profile = parse_resume_with_llm(resume_text)

                st.session_state["resume_text"] = resume_text
                st.session_state["candidate_profile"] = profile.model_dump()
                st.session_state["resume_filename"] = uploaded_file.name
                st.session_state["jobs"] = []
                st.session_state["application_results"] = {}
                st.session_state["recruiter_contacts"] = {}

                st.success("Resume analyzed successfully.")
                st.rerun()

            except Exception as error:
                st.error(f"Resume analysis failed: {error}")


# Section 2: Job search preferences
st.header("2. Job search preferences")

with st.form("preferences_form"):

    previous = st.session_state["preferences"] or {}

    role = st.text_input(
        "Target role or keywords",
        value=previous.get("target_role", ""),
        placeholder="Python Developer, AI Engineer, GenAI Intern",
        help="Separate multiple search keywords with commas.",
    )

    col1, col2 = st.columns(2)

    with col1:
        min_exp = st.number_input(
            "Minimum required experience in years",
            min_value=0,
            max_value=40,
            value=int(previous.get("min_experience", 0)),
            step=1,
        )

    with col2:
        max_exp = st.number_input(
            "Maximum required experience in years",
            min_value=0,
            max_value=40,
            value=int(previous.get("max_experience", 10)),
            step=1,
        )

    job_type_options = [
        "Any", "Internship", "Full time", "Part time", "Contract"
    ]
    previous_job_type = previous.get("job_type", "Any")

    job_type = st.selectbox(
        "Job type",
        job_type_options,
        index=(
            job_type_options.index(previous_job_type)
            if previous_job_type in job_type_options else 0
        ),
    )

    location = st.text_input(
        "Location",
        value=previous.get("location", "India"),
        placeholder="Delhi, Bengaluru, Mumbai",
    )

    work_mode_options = ["Any", "Remote", "Hybrid", "On site"]
    previous_work_mode = previous.get("work_mode", "Any")

    work_mode = st.selectbox(
        "Work mode",
        work_mode_options,
        index=(
            work_mode_options.index(previous_work_mode)
            if previous_work_mode in work_mode_options else 0
        ),
    )

    posting_options = [7, 15, 30]
    previous_posting_days = previous.get("posted_within_days", 7)

    posted_within_days = st.selectbox(
        "Jobs posted within",
        posting_options,
        index=(
            posting_options.index(previous_posting_days)
            if previous_posting_days in posting_options else 0
        ),
        format_func=lambda days: f"Last {days} days",
    )

    job_limit_options = [20, 50, 100]
    previous_job_limit = previous.get("max_jobs", 50)

    max_jobs = st.selectbox(
        "Maximum jobs to retrieve",
        job_limit_options,
        index=(
            job_limit_options.index(previous_job_limit)
            if previous_job_limit in job_limit_options else 1
        ),
    )

    search_submitted = st.form_submit_button(
        "Search jobs",
        type="primary",
    )


# Section 3: Search jobs
if search_submitted:

    if not st.session_state["candidate_profile"]:
        st.error("Please analyze your resume before searching.")

    elif max_exp < min_exp:
        st.error("Maximum experience must be at least the minimum.")

    else:
        preferences = {
            "target_role": role.strip(),
            "min_experience": int(min_exp),
            "max_experience": int(max_exp),
            "job_type": job_type,
            "location": location.strip() or "India",
            "work_mode": work_mode,
            "posted_within_days": posted_within_days,
            "max_jobs": max_jobs,
        }

        st.session_state["preferences"] = preferences

        try:
            with st.spinner("Searching jobs..."):
                jobs = search_naukri_jobs(
                    query=preferences["target_role"],
                    location=preferences["location"],
                    job_type=preferences["job_type"],
                    posted_within_days=preferences["posted_within_days"],
                    max_jobs=preferences["max_jobs"],
                )

            if not jobs:
                st.session_state["jobs"] = []
                st.session_state["application_results"] = {}
                st.session_state["recruiter_contacts"] = {}
                st.warning(
                    "No jobs found. Try different keywords or broaden "
                    "your search preferences."
                )

            else:
                filtered = filter_jobs(
                    jobs,
                    min_experience=preferences["min_experience"],
                    max_experience=preferences["max_experience"],
                    job_type=preferences["job_type"],
                )

                if not filtered:
                    st.info(
                        "No jobs matched all selected filters. "
                        "Showing the original results instead."
                    )
                    filtered = list(jobs)

                selected_location = preferences["location"].lower().strip()

                if selected_location not in {"", "india", "all", "any"}:
                    location_matches = [
                        job for job in filtered
                        if selected_location in str(
                            job.get("location", "")
                        ).lower()
                    ]

                    if location_matches:
                        filtered = location_matches
                    else:
                        st.info(
                            "No exact location matches found. "
                            "Showing jobs from other locations too."
                        )

                if preferences["work_mode"] != "Any":
                    mode_map = {
                        "Remote": ["remote", "work from home", "wfh"],
                        "Hybrid": ["hybrid"],
                        "On site": [
                            "work-from-office",
                            "work from office",
                            "on-site",
                            "onsite",
                            "office",
                        ],
                    }

                    accepted_modes = mode_map[preferences["work_mode"]]

                    mode_matches = [
                        job for job in filtered
                        if any(
                            mode in str(job.get("work_mode", "")).lower()
                            for mode in accepted_modes
                        )
                    ]

                    if mode_matches:
                        filtered = mode_matches
                    else:
                        st.info(
                            "No exact work mode matches found. "
                            "Showing jobs with other work modes too."
                        )

                st.session_state["jobs"] = filtered
                st.session_state["application_results"] = {}
                st.session_state["recruiter_contacts"] = {}

                st.success(f"Found {len(filtered)} job listings.")

        except Exception as error:
            st.error(f"Job search failed: {error}")


# Section 4: Job listings and application generation
jobs = st.session_state["jobs"]

if jobs:

    st.header("3. Job listings")
    st.caption(
        "Analyze each job, generate a tailored application, "
        "and prepare a recruiter email."
    )

    for index, job in enumerate(jobs):

        title = job.get("title") or "Untitled role"
        company = job.get("company") or "Company not specified"
        location_text = job.get("location") or "Location not specified"

        apply_url = job.get("apply_url") or job.get("url") or ""

        description = (
            job.get("description")
            or job.get("jobDescription")
            or job.get("job_description")
            or job.get("summary")
            or ""
        )

        contact_key = str(index)

        with st.container(border=True):

            st.subheader(title)
            st.write(f"**Company:** {company}")
            st.write(f"**Location:** {location_text}")

            exp_min = job.get("experience_min_years")
            exp_max = job.get("experience_max_years")

            if exp_min is not None or exp_max is not None:
                st.write(
                    "**Experience:** "
                    f"{exp_min if exp_min is not None else '?'} to "
                    f"{exp_max if exp_max is not None else '?'} years"
                )
            else:
                st.write("**Experience:** Not specified")

            st.write(
                f"**Work mode:** {job.get('work_mode', 'Not specified')}"
            )
            st.write(
                f"**Job type:** {job.get('job_type', 'Not specified')}"
            )
            st.write(
                f"**Salary:** {job.get('salary_text', 'Not disclosed')}"
            )

            if apply_url:
                st.markdown(f"[View job listing and apply]({apply_url})")

            if description:
                with st.expander("View job description"):
                    st.write(description)
            else:
                st.warning(
                    "A full job description was not returned for this "
                    "listing. The application writer may have limited "
                    "information for tailoring the application."
                )

            # Find recruiter email
            if st.button(
                "Find recruiter email",
                key=f"find_recruiter_{index}",
            ):
                with st.spinner("Searching Hunter for contacts..."):
                    st.session_state["recruiter_contacts"][contact_key] = (
                        find_recruiter_contacts(company)
                    )

            contact_result = st.session_state[
                "recruiter_contacts"
            ].get(contact_key)

            if contact_result:

                if contact_result.get("error"):
                    st.warning(contact_result["error"])

                else:
                    if contact_result.get("domain"):
                        st.caption(
                            f"Company domain: {contact_result['domain']}"
                        )

                    contacts = contact_result.get("contacts", [])

                    if not contacts:
                        st.info(
                            contact_result.get("message")
                            or "No relevant recruiter email was found."
                        )

                    else:
                        st.write("**Possible recruiter contacts**")

                        for contact_index, contact in enumerate(contacts):
                            contact_name = (
                                contact.get("name", "").strip()
                                or "Recruitment contact"
                            )

                            st.write(
                                f"**{contact_name}** · "
                                f"{contact.get('position', 'Role not specified')}"
                            )
                            st.write(f"Email: `{contact['email']}`")
                            st.write(
                                "Verification: "
                                f"{contact.get('verification', 'Not provided')}"
                            )

                            if contact.get("confidence") is not None:
                                st.write(
                                    "Hunter confidence score: "
                                    f"{contact['confidence']}/100"
                                )

                            if contact.get("sources"):
                                source = contact["sources"][0]
                                source_url = (
                                    source.get("uri")
                                    or source.get("url")
                                )
                                if source_url:
                                    st.markdown(
                                        f"[View email source]({source_url})"
                                    )

            # Analyze job fit
            if st.button(
                "Analyze fit against my resume",
                key=f"fit_{index}",
            ):
                try:
                    with st.spinner("Analyzing job fit..."):
                        analyzed = rank_jobs(
                            [job],
                            candidate_profile=(
                                st.session_state["candidate_profile"]
                            ),
                            preferences=(
                                st.session_state["preferences"] or {}
                            ),
                            resume_text=(
                                st.session_state["resume_text"] or ""
                            ),
                        )[0]

                    st.session_state["jobs"][index] = analyzed
                    st.rerun()

                except Exception as error:
                    st.error(f"Job analysis failed: {error}")

            analysis = job.get("fit_analysis")

            if analysis:
                st.write(
                    f"**Resume fit:** {analysis.get('fit_score', 0)}/100"
                )
                st.write(
                    f"**Confidence:** {analysis.get('confidence', 'Unknown')}"
                )
                st.write(
                    "**Recommendation:** "
                    f"{analysis.get('recommendation', 'Review manually')}"
                )
                st.write(analysis.get("summary", ""))

                with st.expander("Detailed match analysis"):
                    st.write("**Matching requirements**")
                    st.write(analysis.get("matching_requirements", []))

                    st.write("**Missing requirements**")
                    st.write(analysis.get("missing_requirements", []))

                    st.write("**Concerns**")
                    st.write(analysis.get("concerns", []))

            # Generate application draft
            if st.button(
                "Generate application draft",
                key=f"draft_{index}",
            ):
                try:
                    with st.spinner(
                        "Generating and verifying application draft..."
                    ):
                        draft = generate_application(
                            resume_text=st.session_state["resume_text"],
                            candidate_profile=(
                                st.session_state["candidate_profile"]
                            ),
                            job=job,
                            personal_details=(
                                st.session_state.get("personal_details", {})
                            ),
                        )

                        result = ApplicationGuard(
                            max_attempts=1
                        ).review(
                            resume_text=st.session_state["resume_text"],
                            candidate_profile=(
                                st.session_state["candidate_profile"]
                            ),
                            job=job,
                            draft=draft,
                        )

                        st.session_state[
                            "application_results"
                        ][contact_key] = result

                    st.rerun()

                except Exception as error:
                    st.error(
                        "Application draft generation failed: "
                        f"{error}"
                    )

            # Display generated application
            result = st.session_state[
                "application_results"
            ].get(contact_key)

            if result:

                draft = result["draft"]
                report = result.get("report")

                st.markdown("**Email subject**")
                st.code(draft.email_subject)

                st.text_area(
                    "Email body",
                    value=draft.email_body,
                    height=220,
                    key=f"email_{index}",
                )

                if report is None:
                    st.error(
                        "Verification could not be completed. This draft "
                        "has not been verified. Please try again before using it."
                    )
                    if result.get("error"):
                        with st.expander("Technical details"):
                            st.code(str(result["error"]))

                elif result.get("safe", False):
                    st.success(
                        "Application verified against your resume and profile. "
                        "Review it once before sending."
                    )
                    if report.verified_facts:
                        with st.expander("Facts supported by your resume"):
                            for fact in report.verified_facts:
                                st.write(f"• {fact}")

                else:
                    st.warning(
                        "This draft has claims that need review. Do not send "
                        "it until you have checked the warnings below."
                    )
                    if result.get("error"):
                        with st.expander("Technical details"):
                            st.code(str(result["error"]))

                if report is not None:
                    for claim in report.unsupported_claims:
                        st.warning(claim)

                    for phrase in report.risky_phrases:
                        st.warning(f"Risky phrase: {phrase}")

                    if report.recommended_changes and not result.get("safe", False):
                        with st.expander("Recommended changes"):
                            for change in report.recommended_changes:
                                st.write(f"• {change}")

                st.info(
                    "These are drafts only. Review the content and attach "
                    "your resume manually in Gmail before sending."
                )

                saved_contacts = st.session_state[
                    "recruiter_contacts"
                ].get(contact_key, {})

                saved_contact_list = saved_contacts.get("contacts", [])

                chosen_emails = [
                    contact["email"]
                    for contact in saved_contact_list
                    if contact.get("email")
                ]

                if chosen_emails:

                    selected_email = st.selectbox(
                        "Recruiter email for this application",
                        chosen_emails,
                        key=f"selected_recruiter_{index}",
                    )

                    current_body = st.session_state.get(
                        f"email_{index}",
                        draft.email_body,
                    )

                    compose_url = gmail_compose_url(
                        selected_email,
                        draft.email_subject,
                        current_body,
                    )

                    st.link_button(
                        "Open email in Gmail",
                        compose_url,
                        key=f"gmail_draft_{index}",
                    )
