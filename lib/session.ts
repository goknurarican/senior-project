// lib/session.ts
import { ExperimentGroup } from "../types/types";

export function assignExperimentGroup(): ExperimentGroup {
  const r = Math.random();

  // "control" ihtimalini tamamen sildik.
  // Çünkü Phase 1 zaten her zaman control.
  // Phase 2 (assignedVariant) her zaman bozuk senaryolardan biri olmalı.
  if (r < 0.33) return "variant_a";
  if (r < 0.66) return "variant_b";
  return "variant_c";
}