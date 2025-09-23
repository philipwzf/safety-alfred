#!/usr/bin/env python3
"""CTL safety evaluation over recorded trajectories."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from .ctl import *  # type: ignore
    from .ctl_parser import *  # type: ignore
    from .trace_to_ctl import trace_file_to_ctl_sequence  # type: ignore
except ImportError:  # pragma: no cover - fallback for script execution
    _PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    from safety_eval.ctl import *  # type: ignore
    from safety_eval.ctl_parser import *  # type: ignore
    from safety_eval.trace_to_ctl import trace_file_to_ctl_sequence  # type: ignore


class SafetyConstraint:
    def __init__(self, operator: str, formula: str, original: str) -> None:
        self.operator = operator
        self.formula = formula
        self.original = original


def parse_constraint(constraint_str: str) -> SafetyConstraint:
    constraint_str = constraint_str.strip()

    normalized_str = re.sub(r"\bNOT\b", "not", constraint_str, flags=re.IGNORECASE)
    normalized_str = re.sub(r"\bAND\b", "and", normalized_str, flags=re.IGNORECASE)
    normalized_str = re.sub(r"\bOR\b", "or", normalized_str, flags=re.IGNORECASE)
    constraint_str = normalized_str

    if constraint_str.startswith("G(") and constraint_str.endswith(")"):
        inner_formula = constraint_str[2:-1]
        return SafetyConstraint("G", inner_formula, constraint_str)
    if constraint_str.startswith("F(") and constraint_str.endswith(")"):
        inner_formula = constraint_str[2:-1]
        return SafetyConstraint("F", inner_formula, constraint_str)
    return SafetyConstraint("", constraint_str, constraint_str)


def convert_safety_constraint_to_ctl(constraint: SafetyConstraint):
    if constraint.operator == "G" and "->" in constraint.formula and "F(" in constraint.formula:
        return handle_device_safety_pattern(constraint)
    if constraint.operator == "G" and "->" in constraint.formula:
        return handle_implication_pattern(constraint)
    if constraint.operator == "G" and constraint.formula.lower().startswith("not("):
        return handle_prohibition_pattern(constraint)
    raise ValueError(f"Unsupported pattern: {constraint.original}")


def handle_device_safety_pattern(constraint: SafetyConstraint):
    parts = constraint.formula.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid implication in: {constraint.formula}")

    antecedent = parts[0].strip()
    consequent = parts[1].strip()

    ant_ctl = parse_atomic_proposition(antecedent)

    if not (consequent.startswith("F(") and consequent.endswith(")")):
        raise ValueError(f"Expected F(...), got: {consequent}")

    inner_consequent = consequent[2:-1]
    cons_ctl = parse_atomic_proposition(inner_consequent)

    implication = CTLOr([CTLNot(ant_ctl), CTLAllEventually(cons_ctl)])
    return CTLAllAlways(implication)


def handle_prohibition_pattern(constraint: SafetyConstraint):
    inner_formula = constraint.formula[4:-1]
    inner_ctl = parse_atomic_proposition(inner_formula)
    return CTLAllAlways(CTLNot(inner_ctl))


def handle_implication_pattern(constraint: SafetyConstraint):
    parts = constraint.formula.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid implication in: {constraint.formula}")

    antecedent = parts[0].strip()
    consequent = parts[1].strip()

    cons_ctl = parse_formula(consequent)
    ant_ctl = parse_formula(antecedent)

    implication = CTLOr([CTLNot(ant_ctl), cons_ctl])
    return CTLAllAlways(implication)


def parse_formula(formula: str):
    formula = formula.strip()

    if formula.startswith("not(") and formula.endswith(")"):
        inner = formula[4:-1]
        return CTLNot(parse_formula(inner))

    if " and " in formula:
        parts = [part.strip() for part in formula.split(" and ")]
        return CTLAnd([parse_formula(part) for part in parts])

    if " or " in formula:
        parts = [part.strip() for part in formula.split(" or ")]
        return CTLOr([parse_formula(part) for part in parts])

    return parse_atomic_proposition(formula)


def parse_atomic_proposition(prop_str: str):
    match = re.match(r"([A-Za-z_]+)\(([^)]+)\)", prop_str.strip())
    if not match:
        raise ValueError(f"Invalid proposition: {prop_str}")

    predicate = match.group(1)
    args_str = match.group(2)
    args = [arg.strip().strip("'\"") for arg in args_str.split(",")]

    predicate_mapping = {
        "IN": "INSIDE",
        "NEXT_TO": "NEXT_TO",
        "ONTOP": "ONTOP",
        "ON": "ON",
        "OFF": "OFF",
        "CLEAN": "CLEAN",
        "DIRTY": "DIRTY",
        "NEAR": "NEAR",
        "CLOSE": "NEAR",
        "INSIDE": "INSIDE",
        "HOLDING": "HOLDING",
        "Holding": "HOLDING",
        "isFilledWithLiquid": "ISFILLEDWITHLIQUID",
        "ISFILLEDWITHLIQUID": "ISFILLEDWITHLIQUID",
    }

    mapped_predicate = predicate_mapping.get(predicate, predicate)

    return CTLPrimitive(Proposition(mapped_predicate, args))


def gather_trace_files(base_dir: Path) -> List[Path]:
    trace_files: List[Path] = []
    for trial_dir in sorted(base_dir.glob("trial_*")):
        if not trial_dir.is_dir():
            continue
        trace_files.extend(sorted(trial_dir.glob("*.json")))
    return trace_files


def load_constraints_from_json(path: Path) -> List[SafetyConstraint]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    constraint_strings: List[str] = []

    def _extend_from_value(value):
        if isinstance(value, list):
            for item in value:
                constraint_strings.append(str(item))
        elif isinstance(value, dict):
            for nested in value.values():
                _extend_from_value(nested)

    if isinstance(payload, list):
        constraint_strings.extend(str(item) for item in payload)
    elif isinstance(payload, dict):
        _extend_from_value(payload)
    else:
        raise ValueError(f"Unsupported constraints JSON format in {path}")

    unique_strings = list(dict.fromkeys(constraint_strings))
    return [parse_constraint(item) for item in unique_strings]


def evaluate_trace(tree: 'TrajectoryTree', constraints: List[SafetyConstraint]) -> Dict[str, List[str]]:
    violations: List[str] = []
    errors: List[str] = []

    for constraint in constraints:
        try:
            ctl_formula = convert_safety_constraint_to_ctl(constraint)
            result = ctl_formula.eval(tree, {})
            if not result.rv:
                violations.append(constraint.original)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{constraint.original} :: {exc}")

    return {"violations": violations, "errors": errors}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CTL safety constraints against trajectory traces")
    parser.add_argument(
        "--task-name",
        required=True,
        help="Task folder under logs/trajectories/train (e.g. pick_and_place_simple-WineBottle-None-Shelf-7)",
    )
    parser.add_argument(
        "--constraints-json",
        default="safety_rules_object.json",
        help="Path to JSON file containing safety constraints",
    )
    parser.add_argument(
        "--constraint-key",
        action="append",
        help="Optional key in constraints JSON to focus on (repeatable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of traces to evaluate",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> bool:
    args = build_arg_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    trace_base = repo_root / "logs" / "trajectories" / "train" / args.task_name

    if not trace_base.exists():
        print(f"✗ Trace directory not found: {trace_base}")
        return False

    trace_files = gather_trace_files(trace_base)
    if args.limit:
        trace_files = trace_files[: args.limit]

    if not trace_files:
        print(f"✗ No trajectory traces found under {trace_base}")
        return False

    constraints_path = Path(args.constraints_json)
    if not constraints_path.is_absolute():
        constraints_path = repo_root / constraints_path

    if not constraints_path.exists():
        print(f"✗ Constraints JSON not found: {constraints_path}")
        return False

    constraints = load_constraints_from_json(constraints_path)
    if not constraints:
        print(f"✗ No constraints extracted from {constraints_path}")
        return False

    parser = CTLParser()
    evaluation_timestamp = datetime.now().isoformat()

    trace_results: List[Dict[str, object]] = []

    for trace_file in trace_files:
        rel_path = trace_file.relative_to(repo_root)
        print(f"Evaluating {rel_path}")

        try:
            ctl_sequence = trace_file_to_ctl_sequence(trace_file)
            tree = parser.to_tree_traj(ctl_sequence)
        except Exception as exc:
            print(f"  ⚠️  Failed to load trace: {exc}")
            trace_results.append({
                "trace": str(rel_path),
                "violations": [],
                "errors": [str(exc)],
            })
            continue

        outcome = evaluate_trace(tree, constraints)
        for violation in outcome["violations"]:
            print(f"  ✗ Violation: {violation}")
        for error in outcome["errors"]:
            print(f"  ⚠️  Error: {error}")
        if not outcome["violations"] and not outcome["errors"]:
            print("  ✓ Safe")

        trace_results.append({
            "trace": str(rel_path),
            **outcome,
        })

    total_traces = len(trace_results)
    num_safe = sum(1 for entry in trace_results if not entry["violations"] and not entry["errors"])
    num_violation = sum(1 for entry in trace_results if entry["violations"])
    num_error = sum(1 for entry in trace_results if entry["errors"])

    print("\n" + "=" * 60)
    print("CTL SAFETY SUMMARY")
    print("=" * 60)
    print(f"Traces evaluated: {total_traces}")
    print(f"Safe traces:      {num_safe}")
    print(f"Violations found: {num_violation}")
    print(f"Evaluation errors:{num_error}")

    summary = {
        "task_name": args.task_name,
        "constraints_json": str(constraints_path.relative_to(repo_root)),
        "constraint_keys": args.constraint_key or [],
        "evaluation_timestamp": evaluation_timestamp,
        "results": trace_results,
    }

    output_path = trace_base / f"ctl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as exc:
        print(f"⚠️  Failed to save summary: {exc}")

    return num_violation == 0 and num_error == 0


if __name__ == "__main__":  # pragma: no cover
    success = main()
    sys.exit(0 if success else 1)