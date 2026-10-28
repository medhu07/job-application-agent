
import json
import time

from groq import Groq
from pydantic import ValidationError

from src.claim_validator import ClaimValidator
from src.application_writer import (
    ApplicationDraft,
    SYSTEM_PROMPT,
    prepare_job_details,
    extract_json_object,
)


class ApplicationGuard:
    def __init__(self, max_attempts=1):
        self.validator = ClaimValidator(max_attempts=3)
        self.client = Groq()
        self.max_attempts = max(0, max_attempts)

    def _correct_draft(
        self,
        resume_text,
        candidate_profile,
        job,
        draft,
        report,
    ):
        payload = {
            "resume": resume_text,
            "candidate_profile": candidate_profile,
            "job": prepare_job_details(job),
            "current_draft": draft.model_dump(),
            "validation_report": report.model_dump(),
            "instructions": [
                "Correct every unsupported claim identified by the validator.",
                "Use only facts supported by the resume or candidate profile.",
                "Do not turn a general skill into a project specific claim.",
                "Do not invent company research, experience, metrics, or achievements.",
                "Remove claims that cannot be supported instead of guessing.",
                "Keep the email concise and the cover letter under 250 words.",
                "Return the complete corrected draft using the required JSON fields.",
            ],
        }

        last_error = None

        # A single correction call is the default. Retry only temporary
        # failures, not factual verification failures.
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    max_completion_tokens=4096,
                    temperature=0,
                )

                choices = getattr(response, "choices", None)

                if not choices:
                    raise ValueError(
                        "The correction model returned no choices."
                    )

                content = getattr(choices[0].message, "content", None)

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        "The correction model returned an empty response."
                    )

                data = extract_json_object(content)
                return ApplicationDraft.model_validate(data)

            except Exception as exc:
                last_error = exc

                status = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(exc, (ValueError, ValidationError))
                    or status == 429
                    or (status is not None and status >= 500)
                    or (
                        status is None
                        and exc.__class__.__module__.startswith(
                            ("httpx", "httpcore", "groq")
                        )
                    )
                )

                if not retryable or attempt == 1:
                    break

                time.sleep(1.5 * (2 ** attempt))

        raise RuntimeError(
            "The application draft could not be corrected."
        ) from last_error

    def review(self, resume_text, candidate_profile, job, draft):
        current_draft = draft
        latest_report = None
        validation_count = 0
        correction_count = 0

        for attempt in range(self.max_attempts + 1):
            try:
                latest_report = self.validator.validate(
                    resume_text=resume_text,
                    candidate_profile=candidate_profile,
                    draft=current_draft,
                )
                validation_count += 1

            except Exception as exc:
                # A failed API call is not evidence that a draft is safe.
                return {
                    "draft": current_draft,
                    "report": None,
                    "safe": False,
                    "attempts": validation_count,
                    "error": str(exc),
                }

            if latest_report.is_safe:
                return {
                    "draft": current_draft,
                    "report": latest_report,
                    "safe": True,
                    "attempts": validation_count,
                    "corrections": correction_count,
                }

            if attempt >= self.max_attempts:
                break

            try:
                current_draft = self._correct_draft(
                    resume_text=resume_text,
                    candidate_profile=candidate_profile,
                    job=job,
                    draft=current_draft,
                    report=latest_report,
                )
                correction_count += 1

            except Exception as exc:
                return {
                    "draft": current_draft,
                    "report": latest_report,
                    "safe": False,
                    "attempts": validation_count,
                    "corrections": correction_count,
                    "error": str(exc),
                }

        return {
            "draft": current_draft,
            "report": latest_report,
            "safe": False,
            "attempts": validation_count,
            "corrections": correction_count,
        }
