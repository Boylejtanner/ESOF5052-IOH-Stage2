"""Print the report's principal aggregate results without loading clinical data."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def rows(relative_path):
    """Read an aggregate CSV while preserving the committed values."""
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    selection = json.loads((ROOT / "configs" / "selected_features.json").read_text(encoding="utf-8"))
    print("STAGE 2: SELECTED PHYSIOLOGICAL CONTEXT")
    print(selection["integration_status"])
    for name in selection["engineered"]:
        print(f"  {name}")

    print("\nGRADIENT-BOOSTING VALIDATION COMPARISON")
    print(f"{'Inputs':42} {'Setting':31} {'AUROC':>8} {'AUPRC':>8}")
    for row in rows("results/bshi_input_eda/validation_comparisons.csv"):
        if row["recipe"] not in {"g0", "ind3", "ind3_multiscale"}:
            continue
        if row["setting"] not in {"pooled_validation", "vitaldb_to_mover_validation"}:
            continue
        print(f"{row['recipe']:42} {row['setting']:31} {float(row['auroc']):8.4f} {float(row['auprc']):8.4f}")

    print("\nRECORDED SIX-EPOCH BSHI INPUT TRIALS")
    print("Implemented seven-plane image and 25-value context; not the selected new context.")
    for row in rows("results/train/trials_20261009_111433.csv"):
        print(f"{row['name']:30} AUROC={float(row['val_auroc']):.4f} AUPRC={float(row['val_auprc']):.4f}")

    print("\nSELECTED OPTIMIZER")
    print(json.dumps(selection["optimizer"], indent=2))
    sweep = rows("results/train/sweep_trials.csv")
    selected = [r for r in sweep if r["name"] in {"adamw_b95_bs128", "adamw_b95_bs128_seed5053"}]
    assert len(selected) == 2, "Expected the two selected-configuration seeds."
    print(f"Two-seed mean AUPRC: {sum(float(r['val_auprc']) for r in selected) / 2:.4f}")
    print("\nSee docs/RESULTS.md for cohort denominators, uncertainty, and evaluation protocols.")


if __name__ == "__main__":
    main()
