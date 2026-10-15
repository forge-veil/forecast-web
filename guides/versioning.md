---
title: Versioning
group: Guides
---

# Versioning

forecast-web follows semantic versioning from 1.0.0:

- **Minor releases** may add models, options, fields and error codes. `ModelId` and `ErrorCode` grow, so avoid exhaustive switches on them. Re-tuned `anomalyThresholds` and changed default formats also ship in minor releases and are listed in the changelog.
- **Major releases** are needed to remove or rename anything, change the default model, or drop a browser.
- Each release downloads each model's weights at a fixed commit, so a weights update reaches you only through a package update.
- onnxruntime-web may be upgraded within 1.x when every parity test still passes.
- The package is ESM only.
- The [API reference](https://vatsalbakshi.com/forecast-web/api/) is built from `main`, so it can show changes before they reach npm. [jsDocs.io](https://www.jsdocs.io/package/forecast-web) has the API of every published version.
- Anything marked experimental (currently `calibrate` and `Forecast.calibration`) may change in a minor release.

See [CHANGELOG.md](https://github.com/forge-veil/forecast-web/blob/main/CHANGELOG.md) for each release.

## License

Apache-2.0. TimesFM is by Google Research and Chronos by Amazon, both Apache-2.0. TabICL is by the Soda team at Inria under BSD-3-Clause. See [`NOTICE`](https://github.com/forge-veil/forecast-web/blob/main/NOTICE).
