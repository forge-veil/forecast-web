import { validateHorizon } from "src/core/validate";

describe("validateHorizon", () => {
  it("accepts 1 through max", () => {
    expect(() => validateHorizon(1, 64)).not.toThrow();
    expect(() => validateHorizon(64, 64)).not.toThrow();
  });

  it("uses the limit it is given", () => {
    expect(() => validateHorizon(65, 64)).toThrow(/limit of 64/);
    expect(() => validateHorizon(200, 256)).not.toThrow();
  });

  it.each([[0], [-1], [1.5], [NaN]])("rejects %s as INVALID_HORIZON", (h) => {
    expect(() => validateHorizon(h, 64)).toThrow(expect.objectContaining({ code: "INVALID_HORIZON" }));
  });
});
