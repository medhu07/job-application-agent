# AI Job Application Agent

An AI powered job application assistant that helps candidates discover relevant opportunities, analyze their resumes, and generate personalized job application emails based on their actual skills and experience.

Built with Python, Streamlit, and Google Gemini, this project aims to make the job application process more efficient while keeping generated content grounded in the candidate's profile.

## Live Demo

## Live Application:https://job-application-agent-7ogx.onrender.com/


## Features

### Resume Analysis
• Extracts text from uploaded resumes in PDF format.

• Processes candidate information to support personalized applications.

### Job Search
• Retrieves job opportunities through the project's job search modules.

• Supports job discovery workflows, including Naukri related functionality.

### AI Powered Application Generation
• Uses Google Gemini to generate personalized application content.

• Creates a relevant email subject and body based on the candidate's resume and the target job description.

• Provides a summary of candidate fit, supporting evidence, and a positioning strategy.

• Identifies missing information and claims that may require verification.

### Claim Validation
• Includes validation components designed to help keep generated application content consistent with the candidate's actual experience.

### Web Interface
• Provides an interactive interface built with Streamlit.

• Brings job discovery, candidate information, and application generation into one workflow.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface |
| Google Gemini API | AI powered application generation |
| Groq API | LLM integration for supported project components |
| PyMuPDF | PDF resume text extraction |
| Pydantic | Structured data validation |
| python docx | Document processing |
| Apify Client | Integration with supported job search workflows |
| python dotenv | Environment variable management |

## Project Structure

```text
job-application-agent/
├── app.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── application_guard.py
│   ├── application_writer.py
│   ├── claim_validator.py
│   ├── job_search.py
│   ├── naukri_jobs.py
│   └── resume_parser.py
├── test_application_writer.py
├── test_claim_validator.py
└── test_jobs.py
```

## How It Works

1. **Upload a resume:** The application extracts relevant candidate information from the resume.
2. **Find a job:** The user discovers or selects a target opportunity.
3. **Analyze the fit:** The system uses candidate information and job requirements to identify relevant skills and evidence.
4. **Generate the application:** Google Gemini creates personalized application content based on the available information.
5. **Validate the content:** Validation components help identify claims that should be checked before sending.
6. **Review and use:** The candidate reviews the generated content before using it.

## Getting Started

### Prerequisites

• Python 3.11 or another Python version compatible with the project's dependencies.

• API credentials for the services used by the enabled features.

### 1. Clone the Repository

```bash
git clone https://github.com/medhu07/job-application-agent.git
cd job-application-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the credentials required by the enabled features.

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
APIFY_API_TOKEN=your_apify_api_token
HUNTER_API_KEY=your_hunter_api_key
```

Only configure the variables required by your current implementation. Never commit your actual `.env` file or expose API credentials publicly.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Running Tests

The repository includes tests for application generation, claim validation, and job search functionality.

Run the test suite using:

```bash
python -m pytest
```

If pytest is not installed, install it in your virtual environment first:

```bash
python -m pip install pytest
```

## Deployment

The application can be deployed to Render using the connected GitHub repository.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Configure the required API credentials through Render's Environment settings rather than storing secrets in the repository.

## Design Principles

• **Personalization:** Generate application content tailored to the specific role.

• **Evidence based writing:** Ground claims in information available from the candidate's resume and profile.

• **Structured output:** Use validated data models to organize the generated application.

• **Human review:** Treat AI generated content as a draft that should be checked before submission.

• **Modular architecture:** Separate resume processing, job search, application generation, and validation into distinct modules.

## Future Improvements

• Reduce application generation latency.

• Improve job matching and relevance ranking.

• Add support for additional resume formats.

• Introduce application tracking and saved drafts.

• Improve error handling and retry behavior.

• Add more comprehensive automated tests.

• Support additional application formats and workflows.

## Disclaimer

This project is a prototype intended to assist with job discovery and application drafting. Generated content may contain errors and should be reviewed before use. Job availability, eligibility, and application requirements should be verified through the relevant employer or recruitment platform.

## Author

**Medhansh Mathur**

• GitHub: https://github.com/medhu07

• LinkedIn: https://www.linkedin.com/in/medhansh-mathur-703734156/
