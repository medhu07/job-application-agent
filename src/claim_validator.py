
import json
import re
import time

from groq import Groq
from pydantic import BaseModel, Field, ValidationError


class ClaimValidationReport(BaseModel):
    is_safe: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    risky_phrases: list[str] = Field(default_factory=list)
    verified_facts: list[str] = Field(default_factory=list)
    recommended_changes: list[str] = Field(default_factory=list)


VALIDATOR_PROMPT = """
You are a strict factual auditor for AI generated job applications.

The original resume and candidate profile are the only evidence of
the candidate's background and abilities. The job description is not
evidence that the candidate possesses a skill.

Flag invented responsibilities, technologies, methods, metrics,
achievements, employment, qualifications, and professional experience.

Do not assume that a project title proves its implementation details.
Do not assume that using a technology proves expertise in it.
Do not flag reasonable summaries supported by the evidence.
Generic statements of personal interest are acceptable.

Audit the email subject, email body, cover letter, fit summary,
key evidence, and positioning strategy.

Return exactly one JSON object with:
is_safe: boolean
unsupported_claims: list of strings
risky_phrases: list of strings
verified_facts: list of strings
recommended_changes: list of strings

Set is_safe to false if material claims are unsupported or important
claims cannot be verified. Explain unsupported claims specifically.
Do not return Markdown fences or text outside the JSON object.
"""


class ClaimValidator:
    def __init__(
        self,
        model="openai/gpt-oss-20b",
        max_attempts=3,
        retry_delay=1.5,
    ):
        self.client = Groq()
        self.model = model
        self.max_attempts = max(1, max_attempts)
        self.retry_delay = max(0.5, retry_delay)

    @staticmethod
    def _draft_to_dict(draft):
        if isinstance(draft, BaseModel):
            return draft.model_dump()

        if isinstance(draft, dict):
            return draft

        raise ValueError(
            "Draft must be a dictionary or a Pydantic model."
        )

    @staticmethod
    def _parse_json(content):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The validator returned an empty response.")

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
            raise ValueError(
                "The validator did not return a JSON object."
            )

        try:
            result = json.loads(content[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The validator returned malformed JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("The validation result must be an object.")

        return result

    def validate(self, resume_text, candidate_profile, draft):
        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text is empty.")

        if not isinstance(candidate_profile, dict):
            raise ValueError("Candidate profile must be a dictionary.")

        payload = {
            "candidate_evidence": {
                "original_resume": resume_text,
                "candidate_profile": candidate_profile,
            },
            "generated_application": self._draft_to_dict(draft),
            "task": (
                "Audit all material factual claims against the original "
                "candidate evidence. Return specific unsupported claims. "
                "Do not use the job description as evidence of skills."
            ),
        }

        last_error = None

        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": VALIDATOR_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    max_completion_tokens=3000,
                    temperature=0,
                )

                choices = getattr(response, "choices", None)

                if not choices:
                    raise ValueError(
                        "The validator returned no response choices."
                    )

                content = getattr(choices[0].message, "content", None)

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        "The validator returned an empty response."
                    )

                result = self._parse_json(content)
                report = ClaimValidationReport.model_validate(result)

                if report.unsupported_claims or report.risky_phrases:
                    report.is_safe = False

                return report

            except Exception as exc:
                last_error = exc

                # Retry malformed or empty output, rate limits, temporary
                # server failures, and network errors. Do not retry
                # permanent authentication or invalid request failures.
                status = getattr(exc, "status_code", None)

                retryable_status = (
                    status == 429
                    or (status is not None and status >= 500)
                )

                retryable_output = isinstance(
                    exc,
                    (ValueError, ValidationError),
                )

                retryable_network = (
                    status is None
                    and not isinstance(exc, (ValueError, ValidationError))
                    and exc.__class__.__module__.startswith(
                        ("httpx", "httpcore", "groq")
                    )
                )

                should_retry = (
                    retryable_status
                    or retryable_output
                    or retryable_network
                )

                if not should_retry or attempt == self.max_attempts - 1:
                    break

                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)

        raise RuntimeError(
            "Claim verification could not complete after "
            f"{self.max_attempts} attempts. "
            "No verified result was produced."
        ) from last_error
