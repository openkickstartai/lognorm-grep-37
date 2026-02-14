# LogNorm

Log quality scanner that detects semantic duplicates, observability blind spots, and level-severity mismatches in Python codebases.

## Problem

`logger.info('user logged in')` and `logger.info('User login successful')` in different modules make grep useless during incidents. Silent `except` blocks swallow errors. `logger.error()` records "avatar updated successfully".

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Scan a directory
python lognorm.py src/

# Scan specific files with JSON output
python lognorm.py app.py handlers.py --format json

# Custom similarity threshold (0-1, default 0.8)
python lognorm.py src/ --threshold 0.7
```

## What It Detects

### Semantic Duplicates
Log messages with high TF-IDF cosine similarity across different locations.

### Blind Spots
`except` blocks with no logging — errors silently swallowed.

### Level-Severity Mismatches
- Severe words (`fail`, `error`, `timeout`) logged at `DEBUG`/`INFO`
- Mild words (`success`, `complete`, `updated`) logged at `ERROR`/`CRITICAL`

### Fingerprint Collisions
Identical messages in multiple locations that prevent tracing to source.

### Missing Context
Log calls in functions with `request_id`, `user_id`, `trace_id` etc. in scope
that forget to include them — making post-incident correlation impossible.

## Output

```
LogNorm Report
========================================
Log calls found:    4
Duplicate groups:   1
Blind spots:        1
Level mismatches:   2
FP collisions:      1
Missing context:    1
```

Exit code `1` when issues found — use in CI to gate PRs.

## Tests

```bash
python -m pytest test_lognorm.py -v
```

## License

MIT
