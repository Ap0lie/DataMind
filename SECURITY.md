# Security Policy

## Supported Version

Security fixes are applied to the current `main` branch. Historical commits and
locally modified deployments are not maintained as separate supported releases.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub private
vulnerability reporting when it is enabled for the repository, or contact the
repository owner privately through their GitHub profile.

Include the affected component, reproduction steps, expected impact, and any
relevant logs with credentials and user data removed. Please do not test against
systems or datasets that you do not own or have permission to access.

## Security Boundaries

- Never commit provider keys, `.env` files, databases, uploaded datasets, model
  caches, report attachments, or browser traces.
- Treat generated SQL, Python, LLM output, uploaded text, and image OCR as
  untrusted input.
- Production deployments must use session authentication, CSRF and Origin
  checks, PostgreSQL, Redis/Celery, and the controlled Python Runner.
- A public GitHub repository is not a production deployment. Deployment secrets
  belong in GitHub Secrets or the target platform's secret manager.
