# Security Policy

AeroNet AI is a local image-classification demonstration. It does not handle
user accounts, payments, or external credentials.

## Security considerations

- The application accepts only still images (`.jpg`, `.jpeg`, `.png`).
- Uploads are limited to `16 MiB` and are processed in memory; they are not
  stored on disk after a prediction completes.
- No secrets or API keys are required. The optional `.env` file is documented as
  a template only.

## Reporting a vulnerability

If you believe you have found a security issue in AeroNet AI, please open a
private issue describing the problem and steps to reproduce. We will review
reproducible reports and respond in a timely manner.

## Qualifying security measures

- Do not commit a real `.env` file or any credentials. The project already
  states that `.env` must never contain secrets.
- Do not rely on the bundled demo dataset for production-grade accuracy.
