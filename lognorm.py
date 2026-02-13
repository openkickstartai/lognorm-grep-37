#!/usr/bin/env python3
"""LogNorm CLI \u2014 Log quality scanner for Python codebases."""
import argparse
import json
import os
import sys

from analyzers import extract, detect_duplicates, detect_mismatches
from formatters import to_sarif, to_text

__version__ = "0.1.0"


def scan_files(paths, threshold=0.8):
    all_calls, all_blindspots = [], []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.py'):
                        _scan_one(os.path.join(dirpath, f), all_calls, all_blindspots)
        elif path.endswith('.py'):
            _scan_one(path, all_calls, all_blindspots)
    dupes = detect_duplicates(all_calls, threshold)
    mismatches = detect_mismatches(all_calls)
    return {'calls': all_calls, 'blindspots': all_blindspots,
            'duplicates': dupes, 'mismatches': mismatches}


def _scan_one(filepath, calls, blindspots):
    try:
        with open(filepath, encoding='utf-8') as f:
            c, b = extract(f.read(), filepath)
        calls.extend(c)
        blindspots.extend(b)
    except (SyntaxError, UnicodeDecodeError):
        return


def results_to_findings(results):
    """Flatten scan results into a list of finding dicts."""
    findings = []
    for group in results['duplicates']:
        for call in group:
            findings.append({
                'ruleId': 'lognorm/semantic-duplicate',
                'message': f'Semantic duplicate: "{call.message}"',
                'file': call.file,
                'line': call.line,
                'severity': 'normal',
            })
    for bs in results['blindspots']:
        findings.append({
            'ruleId': 'lognorm/blind-spot',
            'message': f'Silent {bs.kind}: no logging in exception handler',
            'file': bs.file,
            'line': bs.line,
            'severity': 'high',
        })
    for m in results['mismatches']:
        findings.append({
            'ruleId': 'lognorm/level-mismatch',
            'message': f'Level mismatch: {m.reason} (level={m.log.level}, expected>={m.expected_min})',
            'file': m.log.file,
            'line': m.log.line,
            'severity': 'normal',
        })
    return findings


def build_report(results):
    s = {
        'total_log_calls': len(results['calls']),
        'duplicate_groups': len(results['duplicates']),
        'blind_spots': len(results['blindspots']),
        'level_mismatches': len(results['mismatches']),
    }
    details = {
        'duplicates': [
            [{'file': c.file, 'line': c.line, 'level': c.level,
              'message': c.message} for c in g]
            for g in results['duplicates']
        ],
        'blind_spots': [
            {'file': b.file, 'line': b.line, 'kind': b.kind}
            for b in results['blindspots']
        ],
        'mismatches': [
            {'file': m.log.file, 'line': m.log.line, 'level': m.log.level,
             'message': m.log.message, 'expected_min': m.expected_min,
             'reason': m.reason}
            for m in results['mismatches']
        ],
    }
    return {'summary': s, 'details': details}


def main():
    parser = argparse.ArgumentParser(
        description='LogNorm \u2014 Log quality scanner for Python codebases')
    parser.add_argument('paths', nargs='+', help='Files or directories to scan')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Similarity threshold (0-1, default 0.8)')
    parser.add_argument('--format', choices=['text', 'json', 'sarif'],
                        default='text', help='Output format (default: text)')
    args = parser.parse_args()

    results = scan_files(args.paths, args.threshold)
    findings = results_to_findings(results)

    if args.format == 'sarif':
        sarif = to_sarif(findings, tool_version=__version__)
        print(json.dumps(sarif, indent=2))
    elif args.format == 'json':
        report = build_report(results)
        print(json.dumps(report, indent=2))
    else:
        print(to_text(findings))

    has_issues = (results['duplicates'] or results['blindspots']
                  or results['mismatches'])
    sys.exit(1 if has_issues else 0)


if __name__ == '__main__':
    main()
