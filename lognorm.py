#!/usr/bin/env python3
"""LogNorm CLI — Log quality scanner for Python codebases."""
import argparse
import json
import os
import sys

from analyzers import extract, detect_duplicates, detect_mismatches


def scan_files(paths, threshold=0.8):
    all_calls, all_blindspots = [], []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(root):
                for f in files:
                    if f.endswith('.py'):
                        _scan_one(os.path.join(root, f), all_calls, all_blindspots)
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
             'message': m.log.message, 'reason': m.reason}
            for m in results['mismatches']
        ],
    }
    return {'summary': s, 'details': details}


def print_text(report):
    s = report['summary']
    print("LogNorm Report")
    print("=" * 40)
    print(f"Log calls found:    {s['total_log_calls']}")
    print(f"Duplicate groups:   {s['duplicate_groups']}")
    print(f"Blind spots:        {s['blind_spots']}")
    print(f"Level mismatches:   {s['level_mismatches']}")
    for i, group in enumerate(report['details']['duplicates'], 1):
        print(f"\n\u26a0 Duplicate Group {i}:")
        for c in group:
            print(f"  {c['file']}:{c['line']} [{c['level']}] \"{c['message']}\"")
    for b in report['details']['blind_spots']:
        print(f"\n\U0001f507 Blind spot: {b['file']}:{b['line']} ({b['kind']})")
    for m in report['details']['mismatches']:
        print(f"\n\U0001f500 Mismatch: {m['file']}:{m['line']} [{m['level']}] \"{m['message']}\"")
        print(f"   Reason: {m['reason']}")


def main():
    p = argparse.ArgumentParser(prog='lognorm', description='Log quality scanner')
    p.add_argument('paths', nargs='+', help='Files or directories to scan')
    p.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold')
    p.add_argument('--format', choices=['json', 'text'], default='text', dest='fmt')
    args = p.parse_args()
    results = scan_files(args.paths, args.threshold)
    report = build_report(results)
    if args.fmt == 'json':
        print(json.dumps(report, indent=2))
    else:
        print_text(report)
    total = s = report['summary']
    issues = s['duplicate_groups'] + s['blind_spots'] + s['level_mismatches']
    sys.exit(1 if issues else 0)


if __name__ == '__main__':
    main()
