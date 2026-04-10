#!/usr/bin/env python3
"""
gate_eval.py — GPU pipeline gate condition evaluator
Generated from gate_eval.py.tmpl by generate_pipeline.py

Exit codes:
  0 = all conditions PASS
  1 = one or more conditions FAIL (metrics loaded successfully)
  2 = system error (key missing, NaN, JSON parse failure, file not found)

Metric key format: \\w+ only (alphanumeric + underscore, no spaces)
Supported operators: >=, <=, ==, !=, >, <
Supported logic: AND (OR not supported)
"""
import argparse
import json
import math
import re
import sys

OPERATORS = {
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '>':  lambda a, b: a > b,
    '<':  lambda a, b: a < b,
}

TOKEN_RE = re.compile(r'^(\w+)\s*(>=|<=|==|!=|>|<)\s*([\d.]+)$')


def load_metrics(metrics_arg):
    """Load metrics from inline JSON string or file path."""
    if metrics_arg.strip().startswith('{'):
        return json.loads(metrics_arg)
    with open(metrics_arg) as f:
        return json.load(f)


def evaluate_condition(condition_str, metrics):
    """
    Parse and evaluate 'key>=0.30 AND key2<=0.50' style conditions.
    Returns (passed: bool|None, details: list[str], error: str|None)
      - None passed + error str = exit 2 (system error)
      - bool passed + no error  = exit 0/1
    """
    tokens = [t.strip() for t in condition_str.split('AND')]
    details = []
    for token in tokens:
        m = TOKEN_RE.match(token)
        if not m:
            return None, [], f"Parse error: cannot parse token '{token}'"
        key, op, threshold_str = m.group(1), m.group(2), m.group(3)
        threshold = float(threshold_str)

        if key not in metrics:
            return None, [], f"Missing key: '{key}' not found in metrics"

        val = metrics[key]
        if val is None:
            return None, [], f"Null value: '{key}' is null"
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None, [], f"Non-numeric value: '{key}' = {val!r}"
        if math.isnan(fval):
            return None, [], f"NaN value: '{key}' is NaN"

        result = OPERATORS[op](fval, threshold)
        label = 'PASS' if result else 'FAIL'
        details.append(f"{key}={fval} {op} {threshold}: {label}")

    all_pass = all('PASS' in d for d in details)
    return all_pass, details, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate gate conditions against metrics')
    parser.add_argument('--metrics', required=True,
                        help='JSON file path or inline JSON string (e.g. \'{"rho": 0.35}\')')
    parser.add_argument('--condition', required=True,
                        help='Condition string (e.g. "rho>=0.30 AND auc_prc>=0.40")')
    parser.add_argument('--phase', default='',
                        help='Phase identifier for logging')
    args = parser.parse_args()

    # Load metrics
    try:
        metrics = load_metrics(args.metrics)
    except FileNotFoundError:
        result = {'pass': None, 'metrics': {}, 'phase': args.phase,
                  'message': f"ERROR: file not found: {args.metrics}"}
        print(json.dumps(result))
        sys.exit(2)
    except json.JSONDecodeError as e:
        result = {'pass': None, 'metrics': {}, 'phase': args.phase,
                  'message': f"ERROR: JSON parse failed: {e}"}
        print(json.dumps(result))
        sys.exit(2)

    # Evaluate condition
    passed, details, error = evaluate_condition(args.condition, metrics)

    if error is not None:
        result = {'pass': None, 'metrics': metrics, 'phase': args.phase,
                  'message': f"ERROR: {error}", 'details': []}
        print(json.dumps(result))
        sys.exit(2)

    message = 'All gates passed' if passed else f"Gate failed: {'; '.join(d for d in details if 'FAIL' in d)}"
    result = {
        'pass': passed,
        'metrics': metrics,
        'phase': args.phase,
        'message': message,
        'details': details,
    }
    print(json.dumps(result))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
