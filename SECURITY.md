# Security policy

## Supported versions

Saccade is in early development (0.x). Security fixes are released for the latest version
only. Upgrade with `pip install -U saccade-video`.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/gabe-santana/saccade/security/advisories/new)
(*Security → Report a vulnerability* on the repository page), or by email to
**gsantana.sza@gmail.com** with "Saccade security" in the subject.

Please include:

- a description of the issue and its impact;
- steps or a proof of concept to reproduce it;
- the affected version (`saccade --version`) and platform.

You'll get an acknowledgement within 72 hours and an assessment within a week. Once a fix is
released, the advisory is published, crediting you unless you prefer to stay anonymous.

## Scope and security model

Saccade processes media files locally and, when configured, sends extracted text and images to
an LLM endpoint chosen by the user. These are in scope:

- **Malicious media files.** Crashes, hangs or memory exhaustion triggered by crafted input.
  Decoding is done by FFmpeg through PyAV, so issues in FFmpeg itself should also go
  upstream to FFmpeg and PyAV.
- **Path handling.** Writes outside the cache directory, or path traversal through file names.
- **Credential leaks.** API keys appearing in logs, exceptions, `repr()`, cached files or
  error messages.
- **Unintended network access.** Any connection other than model downloads (forbidden by
  `SACCADE_OFFLINE=1`) and the user's configured LLM.
- **Prompt injection.** Text spoken or shown in a video can try to steer the LLM in `ask()`.
  The exploration tools only read the video being asked about. The only thing they write is
  the fetched images, saved into the cache under names Saccade chooses. They have no
  network access and no access to other files. That limits the impact. Reports that get around
  these limits are welcome.

Out of scope: vulnerabilities in third-party LLM services, and the quality or accuracy of
model output.

## Handling API keys

Saccade reads keys from environment variables or explicit arguments, never writes them to
disk, and hides them from `repr()`. Keep keys out of source control. The examples and docs
read them from `AZURE_AI_API_KEY` / `OPENAI_API_KEY`.
