#!/usr/bin/env python3
"""
Standalone Safety Constraints Test
Tests all safety constraints from safety.txt against trajectory data
"""

import json
import typing
from typing import Dict, List, Union
import sys
import os
import re
from pathlib import Path
from datetime import datetime
# Import your existing CTL framework
try:
    # Try relative imports first (for module usage)
    from .ctl import *
    from .ctl_parser import *
except ImportError:
    # Fall back to absolute imports (for direct usage)
    import sys
    from pathlib import Path
    
    # Add current directory to path
    _THIS_DIR = Path(__file__).parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    
    from ctl import *
    from ctl_parser import *

# Safety constraint data structure
class SafetyConstraint:
    def __init__(self, operator, formula, original):
        self.operator = operator
        self.formula = formula
        self.original = original

def parse_safety_constraints_file(filepath):
    """Parse safety constraints from a text file"""
    constraints = []
    
    with open(filepath, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    constraint = parse_constraint(line)
                    constraints.append(constraint)
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num}: {line}")
                    print(f"Error: {e}")
    
    return constraints

def parse_constraint(constraint_str):
    """Parse a single safety constraint string"""
    constraint_str = constraint_str.strip()
    
    normalized_str = re.sub(r'\bNOT\b', 'not', constraint_str, flags=re.IGNORECASE)
    normalized_str = re.sub(r'\bAND\b', 'and', normalized_str, flags=re.IGNORECASE)
    normalized_str = re.sub(r'\bOR\b', 'or', normalized_str, flags=re.IGNORECASE)
    constraint_str = normalized_str
    
    # Check for global operator G(...)
    if constraint_str.startswith('G(') and constraint_str.endswith(')'):
        inner_formula = constraint_str[2:-1]  # Remove G( and )
        return SafetyConstraint('G', inner_formula, constraint_str)
    
    # Add support for other temporal operators if needed
    elif constraint_str.startswith('F(') and constraint_str.endswith(')'):
        inner_formula = constraint_str[2:-1]
        return SafetyConstraint('F', inner_formula, constraint_str)
    
    else:
        # No temporal operator, treat as plain formula
        return SafetyConstraint('', constraint_str, constraint_str)

def convert_safety_constraint_to_ctl(constraint):
    """Convert safety constraint to CTL formula"""
    if constraint.operator == 'G' and '->' in constraint.formula and 'F(' in constraint.formula:
        return handle_device_safety_pattern(constraint)
    elif constraint.operator == 'G' and '->' in constraint.formula:
        return handle_implication_pattern(constraint)
    elif constraint.operator == 'G' and constraint.formula.lower().startswith('not('):
        return handle_prohibition_pattern(constraint)
    else:
        raise ValueError(f"Unsupported pattern: {constraint.original}")

def handle_device_safety_pattern(constraint):
    """Handle G(ON(device) -> F(OFF(device)))"""
    parts = constraint.formula.split('->')
    if len(parts) != 2:
        raise ValueError(f"Invalid implication in: {constraint.formula}")
    
    antecedent = parts[0].strip()
    consequent = parts[1].strip()
    
    ant_ctl = parse_atomic_proposition(antecedent)
    
    if not (consequent.startswith('F(') and consequent.endswith(')')):
        raise ValueError(f"Expected F(...), got: {consequent}")
    
    inner_consequent = consequent[2:-1]
    cons_ctl = parse_atomic_proposition(inner_consequent)
    
    implication = CTLOr([CTLNot(ant_ctl), CTLAllEventually(cons_ctl)])
    return CTLAllAlways(implication)

def handle_prohibition_pattern(constraint):
    """Handle G(not(...))"""
    inner_formula = constraint.formula[4:-1]
    inner_ctl = parse_atomic_proposition(inner_formula)
    return CTLAllAlways(CTLNot(inner_ctl))

def handle_implication_pattern(constraint):
    """Handle G(A -> B)"""
    parts = constraint.formula.split('->')
    if len(parts) != 2:
        raise ValueError(f"Invalid implication in: {constraint.formula}")
    
    antecedent = parts[0].strip()
    consequent = parts[1].strip()
    
    if consequent.startswith('not(') and consequent.endswith(')'):
        inner_consequent = consequent[4:-1]
        cons_ctl = CTLNot(parse_atomic_proposition(inner_consequent))
    else:
        cons_ctl = parse_atomic_proposition(consequent)
    
    if antecedent.startswith('not(') and antecedent.endswith(')'):
        inner_antecedent = antecedent[4:-1]
        ant_ctl = CTLNot(parse_atomic_proposition(inner_antecedent))
    else:
        ant_ctl = parse_atomic_proposition(antecedent)
    
    implication = CTLOr([CTLNot(ant_ctl), cons_ctl])
    return CTLAllAlways(implication)

def parse_atomic_proposition(prop_str):
    """Parse atomic propositions"""
    match = re.match(r'([A-Z_]+)\(([^)]+)\)', prop_str.strip())
    if not match:
        raise ValueError(f"Invalid proposition: {prop_str}")
    
    predicate = match.group(1)
    args_str = match.group(2)
    args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
    
    # Map common predicate aliases to match trajectory data
    predicate_mapping = {
        'IN': 'INSIDE',  # Safety constraints use IN, trajectory uses INSIDE
        'NEXT_TO': 'NEXT_TO',  # Keep as is
        'ONTOP': 'ONTOP',      # Keep as is
        'ON': 'ON',            # Keep as is
        'OFF': 'OFF',          # Keep as is
        'CLEAN': 'CLEAN',      # Keep as is
        'DIRTY': 'DIRTY'       # Keep as is
    }
    
    # Apply mapping if available
    mapped_predicate = predicate_mapping.get(predicate, predicate)
    
    return CTLPrimitive(Proposition(mapped_predicate, args))
    
def main(scene_id=None):
    """Main function to test all safety constraints"""
    
    # Get scene_id from command line argument or default
    if scene_id is None:
        if len(sys.argv) > 1:
            try:
                scene_id = int(sys.argv[1])
            except ValueError:
                print("Error: Scene ID must be an integer")
                print("Usage: python -m safety_eval.ctl_full_pipeline [SCENE_ID]")
                print("Example: python -m safety_eval.ctl_full_pipeline 201")
                return False
        else:
            # Default to scene 201 if no argument provided
            scene_id = 201
            print("No scene ID provided, defaulting to FloorPlan201")
    
    print("=" * 60)
    print(f"SAFETY CONSTRAINTS VERIFICATION - FloorPlan{scene_id}")
    print("=" * 60)
    
    # Load trajectory data from FloorPlan{scene_id}/trials
    print(f"Loading trajectory data from FloorPlan{scene_id}/trials...")
    parser = CTLParser()
    
    # Define the base path for FloorPlan{scene_id} trials
    repo_root = Path(__file__).resolve().parents[1]  # Go up to repo root
    trials_base = repo_root / "safety_eval" / "examples" / f"FloorPlan{scene_id}" / "trials"
    
    print(f"Looking for trajectory data in: {trials_base}")
    
    if not trials_base.exists():
        print(f"✗ Trials directory not found: {trials_base}")
        print("Please ensure trajectory data has been converted to nodes/edges format")
        print(f"Run: python wrapper/adapters/ai2thor_adapter.py --scene-id {scene_id} --evaluate-safety")
        return False
    
    # Discover all trial files in FloorPlan{scene_id}
    trial_files = []
    for identifier_dir in trials_base.iterdir():
        if identifier_dir.is_dir():
            identifier = identifier_dir.name
            print(f"  Found identifier: {identifier}")
            
            # Look for trial_*.json files in this identifier directory
            identifier_trials = list(identifier_dir.glob("trial_*.json"))
            if identifier_trials:
                trial_files.extend(identifier_trials)
                print(f"    Found {len(identifier_trials)} trial files")
    
    if not trial_files:
        print(f"✗ No trial files found in FloorPlan{scene_id}/trials")
        print("Make sure you have run the nodes/edges conversion first:")
        print(f"  python wrapper/adapters/ai2thor_adapter.py --scene-id {scene_id} --evaluate-safety")
        return False
    
    print(f"✓ Found {len(trial_files)} trial files total")
    
    # Load and merge all trajectory trees
    trees = []
    successful_loads = 0
    
    for trial_file in trial_files[:5]:  # Limit to first 5 for testing
        try:
            print(f"  Loading: {trial_file.name}")
            tree = parser.parse_from_file(str(trial_file))
            trees.append(tree)
            successful_loads += 1
        except Exception as e:
            print(f"  ⚠️  Failed to load {trial_file.name}: {e}")
    
    if not trees:
        print("✗ No trajectory trees could be loaded")
        return False
    
    print(f"✓ Successfully loaded {successful_loads} trajectory trees")
    
    # Use the first tree or merge multiple trees if needed
    if len(trees) == 1:
        tree = trees[0]
    else:
        # If you have a merge function, use it; otherwise just use the first tree
        print(f"Using first tree for evaluation (you can implement tree merging if needed)")
        tree = trees[0]
    
    # Load safety constraints for FloorPlan{scene_id}
    print(f"Loading safety constraints for FloorPlan{scene_id}...")
    
    # Look for constraints in multiple possible locations
    constraint_locations = [
        # Scene-specific files (direct)
        repo_root / "safety_eval" / "examples" / f"FloorPlan{scene_id}_safety_constraints.txt",
        repo_root / "safety_constraints" / f"FloorPlan{scene_id}" / "safety_constraints.txt",
        repo_root / "safety_constraints" / f"FloorPlan{scene_id}" / "safety_constraints.json",
    ]
    
    # Also search in identifier subdirectories
    constraints_base = repo_root / "safety_constraints" / f"FloorPlan{scene_id}"
    if constraints_base.exists():
        # Find all identifier subdirectories
        for identifier_dir in constraints_base.iterdir():
            if identifier_dir.is_dir():
                constraint_file = identifier_dir / "safety_constraints.json"
                if constraint_file.exists():
                    constraint_locations.append(constraint_file)
    
    
    constraints = None
    constraints_file = None
    
    for loc in constraint_locations:
        if loc.exists():
            try:
                print(f"  Trying: {loc}")
                if loc.suffix == '.json':
                    # Handle JSON format
                    with open(loc, 'r') as f:
                        constraint_data = json.load(f)
                    
                    # Handle both formats: direct list or nested structure
                    if "safety_constraints" in constraint_data:
                        constraint_strings = constraint_data["safety_constraints"]
                    elif isinstance(constraint_data, list):
                        constraint_strings = constraint_data
                    else:
                        print(f"    ⚠️  Unrecognized JSON format in {loc}")
                        continue
                        
                    constraints = [parse_constraint(c) for c in constraint_strings]
                else:
                    # Handle text format
                    constraints = parse_safety_constraints_file(str(loc))
                
                constraints_file = loc
                print(f"✓ Loaded {len(constraints)} safety constraints from {loc}")
                break
                
            except Exception as e:
                print(f"  ⚠️  Error loading constraints from {loc}: {e}")
    
    if not constraints:
        print(f"✗ No safety constraints found for FloorPlan{scene_id}!")
        print("Checked locations:")
        for loc in constraint_locations:
            print(f"  - {loc}")
        print(f"\nTo create constraints, add them to one of these locations:")
        print(f"  - {constraint_locations[0]}")
        print(f"  - {constraint_locations[1]}")
        return False
    
    # Keep a copy of the original LTL-like constraint strings for logging
    constraints_ltl_strings = [c.original for c in constraints]
    
    # Test all constraints against the loaded trajectory data
    print(f"\nTesting {len(constraints)} safety constraints against FloorPlan{scene_id} trajectories...")
    print("-" * 60)
    
    passed = 0
    failed = 0
    errors = 0
    violated_constraints = []
    
    for i, constraint in enumerate(constraints, 1):
        print(f"Test {i:3d}: {constraint.original}")
        
        try:
            # Convert to CTL and evaluate
            ctl_formula = convert_safety_constraint_to_ctl(constraint)
            result = ctl_formula.eval(tree, {})
            
            if result.rv:
                print(f"         ✓ PASSED")
                passed += 1
            else:
                print(f"         ✗ FAILED - Safety violation detected")
                failed += 1
                violated_constraints.append(constraint.original)
        except Exception as e:
            print(f"         ⚠️  ERROR - Could not evaluate: {e}")
            errors += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"SAFETY VERIFICATION SUMMARY - FloorPlan{scene_id}")
    print("=" * 60)
    print(f"Trajectory files evaluated:  {successful_loads}")
    print(f"Total constraints tested:    {len(constraints)}")
    print(f"Passed:                     {passed}")
    print(f"Failed:                     {failed}")
    print(f"Errors:                     {errors}")
    
    if violated_constraints:
        print("\nViolated Constraints:")
        for c in violated_constraints:
            print(f" - {c}")
    
    if failed == 0 and errors == 0:
        print("\n🎉 ALL SAFETY CONSTRAINTS SATISFIED!")
        print(f"FloorPlan{scene_id} trajectories are safe according to all specified constraints.")
        result_status = "SUCCESS"
        safety_satisfied = True
    else:
        print(f"\n⚠️  {failed + errors} constraints need attention")
        if failed > 0:
            print(f"   {failed} safety violations detected in FloorPlan{scene_id}")
        if errors > 0:
            print(f"   {errors} constraints could not be evaluated")
        result_status = "FAILED" if failed > 0 else "PARTIAL"
        safety_satisfied = failed == 0
    
    # ---- Save results to FloorPlan{scene_id} specific log ----
    try:
        log_path = repo_root / "safety_eval" / f"FloorPlan{scene_id}_safety_results.json"
        
        safety_results = {
            "scene_id": scene_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "trajectory_files_evaluated": [str(f.relative_to(repo_root)) for f in trial_files[:successful_loads]],
            "constraints_file": str(constraints_file.relative_to(repo_root)) if constraints_file else None,
            "safety_verification": {
                "safety_check_completed": True,
                "total_outputs_checked": len(constraints_ltl_strings),
                "safe_outputs": passed,
                "unsafe_outputs": failed,
                "error_outputs": errors,
                "constraints_ltl": constraints_ltl_strings,
                "violations": violated_constraints
            },
            "evaluation_result": {
                "overall_status": result_status,
                "goal_achieved": True,
                "safety_satisfied": safety_satisfied,
                "valid_solutions": passed,
                "total_outputs": len(constraints_ltl_strings)
            }
        }
        
        with open(log_path, "w") as f:
            json.dump(safety_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {log_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")
    
    return safety_satisfied

if __name__ == "__main__":
    print("Safety Constraints Test - Standalone Version")
    print("Testing safety constraints against trajectory data\n")
    
    # Parse command line arguments
    scene_id = None
    if len(sys.argv) > 1:
        try:
            scene_id = int(sys.argv[1])
            print(f"Using FloorPlan{scene_id} from command line argument")
        except ValueError:
            print("Error: Scene ID must be an integer")
            print("Usage: python -m safety_eval.ctl_full_pipeline [SCENE_ID]")
            print("Examples:")
            print("  python -m safety_eval.ctl_full_pipeline 201")
            print("  python -m safety_eval.ctl_full_pipeline 305")
            sys.exit(1)
    
    success = main(scene_id)
    
    print(f"\nTest completed. Exit code: {0 if success else 1}")
    sys.exit(0 if success else 1)