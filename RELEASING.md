# Releasing

| Workflow | Trigger | Publishes |
|---|---|---|
| `ci.yml` | every push to `main` and every pull request | nothing; typecheck, tests, build, pack check |
| `release-weights.yml` | manual (Actions tab, "Run workflow") | ONNX weights to the Hugging Face Hub |
| `release-npm.yml` | pushing a `v*` tag | the package to npm |

Release the weights before the first npm release, since the package downloads them from the Hub by default.

## One-time setup

1. Push the repo to GitHub.
2. Create two environments under Settings, Environments: `huggingface` and `npm`. Add required reviewers to either if you want a manual approval before publishing.
3. In the `huggingface` environment, add the secret `HF_TOKEN`: a Hugging Face access token with write access to the target repo.
4. For the first npm release only, npm cannot set up trusted publishing yet, because the package does not exist. In the `npm` environment, add the secret `NPM_TOKEN`: a granular npm access token with publish rights and a short expiry.
5. After the first npm release succeeds, open the package settings on npmjs.com and add a trusted publisher: GitHub Actions, this repository, workflow `release-npm.yml`, environment `npm`. Then delete the `NPM_TOKEN` secret. Every later release publishes through trusted publishing with provenance and no stored token.

## Releasing weights

Run "Release weights" from the Actions tab. It:

1. Downloads Google's TimesFM 2.5 checkpoint at the revision pinned in `tools/export_onnx.py`.
2. Exports the fp32 graph and builds the q8 and fp16 variants.
3. Generates reference forecasts with the official Python library.
4. Runs the full test suite, including parity for every format. Nothing uploads if it fails.
5. Uploads `build/hub` to the Hub in one commit, removing weight files that are no longer built, and prints the new revision in the run summary.

Tick "dry run" to build and verify without uploading. The run takes roughly 15 to 30 minutes, mostly the checkpoint download and the parity suite.

## Releasing the package

1. Bump `version` in `package.json` and commit.
2. Tag and push: `git tag v0.1.0 && git push origin main v0.1.0`.

The workflow refuses to publish if the tag and `package.json` version differ, and it reruns typecheck, tests and build first.
