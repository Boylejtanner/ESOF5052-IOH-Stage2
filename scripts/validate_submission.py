"""Check submission structure, notebook hygiene, and aggregate-only result files."""
# Check source syntax, notebook output hygiene, and report dependencies.
from pathlib import Path
import ast
import csv
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = {"case_id", "patient_id", "patient_key", "window_id", "subjectid", "subject_id", "mrn", "log_id", "pid"}


def main():
    """Validate the source package without running data processing or training."""
    problems = []
    modules = list((ROOT / "notebooks" / "hemodynamics").glob("*.py"))
    for path in modules + list((ROOT / "scripts").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"Invalid Python syntax: {path.relative_to(ROOT)}: {exc}")

    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                problems.append(f"Saved notebook execution output: {path.name}, cell {index}")
            text = "".join(cell.get("source", []))
            if re.search(r"[A-Z]:[\\/]Users[\\/]|G:[\\/]My Drive", text, re.IGNORECASE):
                problems.append(f"User-specific notebook path: {path.name}, cell {index}")

    tables = list((ROOT / "results").rglob("*.csv"))
    for path in tables:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        if IDENTIFIERS & {c.lower() for c in columns}:
            problems.append(f"Identifier-level result table: {path.relative_to(ROOT)}")

    report = (ROOT / "report" / "main.tex").read_text(encoding="utf-8")
    if r"\includegraphics" in report:
        problems.append("The report has external figure dependencies.")
    if report.count(r"\begin{figure*}") != 14:
        problems.append("The report must retain all 14 full-width vector figures.")
    labels = re.findall(r"\\label\{([^}]+)\}", report)
    references = re.findall(r"\\ref\{([^}]+)\}", report)
    if len(labels) != len(set(labels)) or not set(references) <= set(labels):
        problems.append("The report has duplicate labels or unresolved source references.")

    selected = json.loads((ROOT / "configs" / "selected_features.json").read_text())
    if len(selected["engineered"]) != 8 or selected["planned_context_values"] != 21:
        problems.append("Selected context dimensions do not match the report.")
    if "not yet" not in selected["integration_status"]:
        problems.append("The selected-context integration status must remain explicit.")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Submission checks passed: {len(modules)} Python modules/tests, {len(notebooks)} clean notebooks, {len(tables)} aggregate CSVs, and 14 embedded figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
