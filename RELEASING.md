# Releasing

Saccade is published to PyPI as **`saccade-video`**; it is imported as `saccade`. The plain
name `saccade` is taken on PyPI by an unrelated project.

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
with PyPI **Trusted Publishing**. GitHub proves its identity to PyPI directly, so no API
token is stored anywhere.

## One-time setup

1. **PyPI.** Sign in at https://pypi.org, go to *Your account → Publishing*, then
   *Add a new pending publisher*, and enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `saccade-video` |
   | Owner | `gabe-santana` |
   | Repository name | `saccade` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

2. **TestPyPI** (optional, for rehearsals). Do the same at https://test.pypi.org, with
   environment name `testpypi`.

3. **GitHub** (optional). The environments `pypi` and `testpypi` are created automatically on
   first use. To require your approval before every upload, add yourself as a *required
   reviewer* on `pypi` under *Settings → Environments*. For private repositories that needs
   a paid GitHub plan.

## Releasing a version

1. Bump the version in `src/saccade/_version.py`. That is the only place it is defined;
   `pyproject.toml` reads it from there.
2. Commit and push to `main`, and wait for CI to pass.
3. Tag the commit and push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

Pushing the tag runs the workflow:

1. lint, type-check and run the tests on Linux, Windows and macOS;
2. check that the tag matches the package version;
3. build the sdist and wheel, and run `twine check`;
4. install the wheel in clean environments and run `saccade --version`;
5. upload to PyPI;
6. create the GitHub release, with the built files attached.

If the upload fails because the PyPI publisher isn't configured yet, configure it as
described above, then use *Re-run failed jobs* on the workflow run. There is no need to
re-tag.

## Rehearsing on TestPyPI

Under *Actions → Publish to PyPI → Run workflow*, choose `testpypi`. Then:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ saccade-video
```

PyPI never accepts the same version twice, on PyPI or TestPyPI. Bump the version for every
upload.

## Building locally

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```
