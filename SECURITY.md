# Security

## Model files

forecast-web downloads model weights over HTTPS from the Hugging Face Hub at a fixed commit per release. The size and SHA-256 of every published weight file at that commit are built into the package (`src/pins.json`), and each download is checked against them before it is used or cached; a mismatch fails the load with `LOAD_FAILED`. So the published weights a release loads are exactly the files it was tested with, whatever the Hub or the network serves. The package itself is published from GitHub Actions with npm provenance.

Other sources are checked against their own `config.json` instead, which catches corrupted or truncated downloads but cannot prove where the files came from. That applies to a custom `weights` repo or URL, and to the published repo at a revision other than the pinned one (such as `revision: "main"`). The check is skipped when that `config.json` lists no hashes, and on insecure pages where the browser offers no Web Crypto (for example plain `http://` on a LAN address).

To control the source entirely, self-host the weights (see "Self-hosting the weights" in the README) and pass `weights` and `wasmPaths`.

Everything runs in the visitor's browser. The package sends no data anywhere; the only requests are for model files and onnxruntime's WASM binary.

## Reporting a vulnerability

Please report it privately through GitHub's "Report a vulnerability" button on the repository's Security tab, not in a public issue.
