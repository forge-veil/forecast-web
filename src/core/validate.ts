import { ForecastError } from "src/types";

/** Throws unless `horizon` is an integer from 1 to `max`. */
export function validateHorizon(horizon: number, max: number): void {
  if (!Number.isInteger(horizon) || horizon < 1)
    throw new ForecastError("INVALID_HORIZON", `horizon must be a positive integer, got ${horizon}`);
  if (horizon > max)
    throw new ForecastError("HORIZON_TOO_LONG", `horizon ${horizon} exceeds the model limit of ${max}`);
}
