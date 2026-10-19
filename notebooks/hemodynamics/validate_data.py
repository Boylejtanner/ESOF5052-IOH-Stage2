"""Verify persisted signals; sample success never implies cohort completeness."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .canonical import assert_patient_independence, atomic_json, load_case, verify_case


def validate_outputs(root):
    root = Path(root)
    results = {}
    representatives = {}
    for dataset, selected_name, case_column in (("vitaldb", "usable_cohort", "caseid"), ("mover", "mover_selected_cohort", "case_id")):
        selected_path = root / "data" / "manifests" / f"{selected_name}.parquet"
        selected = pd.read_parquet(selected_path) if selected_path.exists() else pd.DataFrame()
        expected = set(selected[case_column].astype(str)) if len(selected) else set()
        completed, failed = [], []
        directory = root / "data" / "parquet" / dataset
        for metadata_path in sorted(directory.glob("case_id=*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not verify_case(root, metadata):
                    raise ValueError("Hash/readback verification failed")
                if dataset not in representatives:
                    case = load_case(root, dataset, metadata["case_id"])
                    times = case["waveform_time_sec"]
                    expected_fs = metadata.get("art_final_fs", metadata.get("target_waveform_fs", 100))
                    if len(times) > 1 and not np.allclose(np.diff(times), 1 / expected_fs, atol=1e-7, rtol=1e-7):
                        raise ValueError("Waveform timestamp cadence does not match declared sample rate")
                    if any(not np.isfinite(case[c]).any() for c in ("ART", "ECG", "PPG", "MAP")):
                        raise ValueError("Required stream is missing")
                    representatives[dataset] = {"case_id": metadata["case_id"], "patient_id": metadata.get("patient_id"),
                                                "split": metadata.get("split"), "waveform_rows": len(times), "map_rows": len(case["MAP"])}
                completed.append(metadata)
            except Exception as error:
                failed.append({"metadata": str(metadata_path.relative_to(root)), "error": str(error)})
        valid_ids = {m["case_id"] for m in completed}
        split_path = root / "data" / "manifests" / f"{dataset}_patient_splits.parquet"
        overlaps = None
        if split_path.exists():
            overlaps = assert_patient_independence(pd.read_parquet(split_path))
        patient_assignments_valid = False
        if len(completed) and split_path.exists():
            splits = pd.read_parquet(split_path)
            mapping = dict(zip(splits.patient_id.astype(str), splits.split))
            patient_assignments_valid = all(mapping.get(str(m.get("patient_id"))) == m.get("split") for m in completed)
        results[dataset] = {"selected_cases": len(expected) if selected_path.exists() else None,
                            "valid_completed_cases": len(completed), "remaining_selected_cases": len(expected - valid_ids) if selected_path.exists() else None,
                            "failed_outputs": failed, "processed_bytes": sum(m["processed_bytes"] for m in completed),
                            "patient_split_intersections": overlaps, "output_patient_assignments_valid": patient_assignments_valid,
                            "ready_for_ml_bshi": bool(expected and expected.issubset(valid_ids) and not failed and patient_assignments_valid and overlaps is not None and not any(overlaps.values()))}
    report = {"datasets": results, "representative_loads": representatives,
              "actual_cross_dataset_loader_verified": "vitaldb" in representatives and "mover" in representatives,
              "both_complete_and_ready": all(r["ready_for_ml_bshi"] for r in results.values()),
              "note": "Pilot/diagnostic outputs are real signals, not a claim of full cohort readiness. No forecast windows or BSHI generated."}
    atomic_json(root / "results" / "canonical_dataset_validation.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(validate_outputs(parser.parse_args().root), indent=2))
