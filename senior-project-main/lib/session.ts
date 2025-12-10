// lib/session.ts
import { ExperimentGroup } from "../types/types";

export function assignExperimentGroup(): ExperimentGroup {
  const r = Math.random();
  if (r < 0.25) return "control";
  if (r < 0.5) return "variant_a";
  if (r < 0.75) return "variant_b";
  return "variant_c";
}