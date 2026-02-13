"""Tests for LogNorm analyzers."""
import pytest
from analyzers import extract, detect_duplicates, detect_mismatches, LogCall

SAMPLE = '''
import logging
logger = logging.getLogger(__name__)

def login(user):
    logger.info("user logged in")

def authenticate(user):
    logger.info("User login successful")

def process_payment(amount):
    try:
        charge(amount)
    except Exception:
        return False

def handle_error():
    logger.debug("Connection timeout failed")

def update_avatar(user, path):
    logger.error("User updated avatar successfully")
'''


def test_extract_finds_all_log_calls():
    calls, _ = extract(SAMPLE, 'app.py')
    assert len(calls) == 4
    assert calls[0].func == 'login'
    assert calls[0].level == 'info'
    assert calls[0].message == 'user logged in'
    assert all(c.fingerprint for c in calls)


def test_blind_spot_silent_except():
    _, blindspots = extract(SAMPLE, 'app.py')
    assert len(blindspots) == 1
    assert blindspots[0].kind == 'silent_except'
    assert blindspots[0].file == 'app.py'


def test_no_blind_spot_when_except_has_log():
    code = '''
import logging
logger = logging.getLogger(__name__)
def f():
    try:
        x()
    except Exception:
        logger.error("something broke")
'''
    _, blindspots = extract(code, 'ok.py')
    assert len(blindspots) == 0


def test_level_severity_mismatch_severe_at_debug():
    calls, _ = extract(SAMPLE, 'app.py')
    mismatches = detect_mismatches(calls)
    debug_mismatches = [m for m in mismatches if m.log.level == 'debug']
    assert len(debug_mismatches) == 1
    assert 'severe' in debug_mismatches[0].reason


def test_level_severity_mismatch_mild_at_error():
    calls, _ = extract(SAMPLE, 'app.py')
    mismatches = detect_mismatches(calls)
    error_mismatches = [m for m in mismatches if m.log.level == 'error']
    assert len(error_mismatches) == 1
    assert 'mild' in error_mismatches[0].reason


def test_duplicate_detection_clusters_similar():
    calls = [
        LogCall('a.py', 1, 'f', 'info', 'user logged in', 'a1'),
        LogCall('b.py', 2, 'g', 'info', 'user logged in ok', 'b1'),
        LogCall('c.py', 3, 'h', 'error', 'payment processed', 'c1'),
    ]
    groups = detect_duplicates(calls, threshold=0.4)
    assert len(groups) >= 1
    msgs = {c.message for c in groups[0]}
    assert 'user logged in' in msgs
    assert 'user logged in ok' in msgs


def test_duplicate_detection_no_false_positive():
    calls = [
        LogCall('a.py', 1, 'f', 'info', 'user logged in', 'a1'),
        LogCall('b.py', 2, 'g', 'error', 'payment processed', 'b1'),
    ]
    groups = detect_duplicates(calls, threshold=0.8)
    assert len(groups) == 0


def test_fingerprint_uniqueness():
    calls, _ = extract(SAMPLE, 'app.py')
    fps = [c.fingerprint for c in calls]
    assert len(fps) == len(set(fps))


def test_fstring_extraction():
    code = '''
import logging
logger = logging.getLogger(__name__)
def greet(name):
    logger.info(f"hello {name} welcome")
'''
    calls, _ = extract(code, 'f.py')
    assert len(calls) == 1
    assert 'hello' in calls[0].message
    assert '{}' in calls[0].message


# ---------- SARIF formatter tests ----------

import json
import jsonschema
from formatters import to_sarif, to_text

# Minimal SARIF 2.1.0 structural schema for validation
_SARIF_SCHEMA = {
    "type": "object",
    "required": ["$schema", "version", "runs"],
    "properties": {
        "version": {"type": "string", "const": "2.1.0"},
        "$schema": {"type": "string"},
        "runs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["tool", "results"],
                "properties": {
                    "tool": {
                        "type": "object",
                        "required": ["driver"],
                        "properties": {
                            "driver": {
                                "type": "object",
                                "required": ["name", "version", "rules"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "version": {"type": "string"},
                                    "rules": {"type": "array"},
                                },
                            }
                        },
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["ruleId", "level", "message", "locations"],
                            "properties": {
                                "ruleId": {"type": "string"},
                                "level": {
                                    "type": "string",
                                    "enum": ["error", "warning", "note", "none"],
                                },
                                "message": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {"text": {"type": "string"}},
                                },
                                "locations": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "required": ["physicalLocation"],
                                        "properties": {
                                            "physicalLocation": {
                                                "type": "object",
                                                "required": [
                                                    "artifactLocation",
                                                    "region",
                                                ],
                                                "properties": {
                                                    "artifactLocation": {
                                                        "type": "object",
                                                        "required": ["uri"],
                                                        "properties": {
                                                            "uri": {"type": "string"}
                                                        },
                                                    },
                                                    "region": {
                                                        "type": "object",
                                                        "required": ["startLine"],
                                                        "properties": {
                                                            "startLine": {
                                                                "type": "integer",
                                                                "minimum": 1,
                                                            }
                                                        },
                                                    },
                                                },
                                            }
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


MOCK_FINDINGS = [
    {
        "ruleId": "lognorm/semantic-duplicate",
        "message": 'Semantic duplicate: "user logged in"',
        "file": "app.py",
        "line": 10,
        "severity": "normal",
    },
    {
        "ruleId": "lognorm/semantic-duplicate",
        "message": 'Semantic duplicate: "User login successful"',
        "file": "auth.py",
        "line": 22,
        "severity": "normal",
    },
    {
        "ruleId": "lognorm/blind-spot",
        "message": "Silent silent_except: no logging in exception handler",
        "file": "payments.py",
        "line": 45,
        "severity": "high",
    },
    {
        "ruleId": "lognorm/level-mismatch",
        "message": "Level mismatch: severe words at low level (level=debug, expected>=warning)",
        "file": "handlers.py",
        "line": 30,
        "severity": "low",
    },
    {
        "ruleId": "lognorm/level-mismatch",
        "message": "Level mismatch: mild words at high level (level=error, expected<=info)",
        "file": "views.py",
        "line": 88,
        "severity": "normal",
    },
]


def test_sarif_validates_against_schema():
    """SARIF output must pass jsonschema validation."""
    sarif = to_sarif(MOCK_FINDINGS, tool_version="0.1.0")
    jsonschema.validate(instance=sarif, schema=_SARIF_SCHEMA)


def test_sarif_has_correct_tool_name():
    sarif = to_sarif(MOCK_FINDINGS)
    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["name"] == "LogNorm"


def test_sarif_version_is_2_1_0():
    sarif = to_sarif(MOCK_FINDINGS)
    assert sarif["version"] == "2.1.0"
    assert "sarif-schema-2.1.0" in sarif["$schema"]


def test_sarif_contains_all_three_rule_ids():
    sarif = to_sarif(MOCK_FINDINGS)
    results = sarif["runs"][0]["results"]
    rule_ids = {r["ruleId"] for r in results}
    assert "lognorm/semantic-duplicate" in rule_ids
    assert "lognorm/blind-spot" in rule_ids
    assert "lognorm/level-mismatch" in rule_ids


def test_sarif_result_count_matches_findings():
    sarif = to_sarif(MOCK_FINDINGS)
    results = sarif["runs"][0]["results"]
    assert len(results) == len(MOCK_FINDINGS)


def test_sarif_level_mapping_high_to_error():
    sarif = to_sarif(MOCK_FINDINGS)
    results = sarif["runs"][0]["results"]
    blind_spot_results = [r for r in results if r["ruleId"] == "lognorm/blind-spot"]
    assert all(r["level"] == "error" for r in blind_spot_results)


def test_sarif_level_mapping_normal_to_warning():
    sarif = to_sarif(MOCK_FINDINGS)
    results = sarif["runs"][0]["results"]
    dupe_results = [r for r in results if r["ruleId"] == "lognorm/semantic-duplicate"]
    assert all(r["level"] == "warning" for r in dupe_results)


def test_sarif_level_mapping_low_to_note():
    findings = [
        {
            "ruleId": "lognorm/level-mismatch",
            "message": "test",
            "file": "a.py",
            "line": 1,
            "severity": "low",
        }
    ]
    sarif = to_sarif(findings)
    assert sarif["runs"][0]["results"][0]["level"] == "note"


def test_sarif_physical_location_fields():
    sarif = to_sarif(MOCK_FINDINGS)
    loc = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "app.py"
    assert loc["region"]["startLine"] == 10


def test_sarif_rules_only_include_used():
    findings = [
        {
            "ruleId": "lognorm/blind-spot",
            "message": "test",
            "file": "a.py",
            "line": 1,
            "severity": "high",
        }
    ]
    sarif = to_sarif(findings)
    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    assert len(rules) == 1
    assert rules[0]["id"] == "lognorm/blind-spot"


def test_sarif_empty_findings():
    sarif = to_sarif([])
    assert sarif["runs"][0]["results"] == []
    assert sarif["version"] == "2.1.0"


def test_sarif_is_valid_json_roundtrip():
    sarif = to_sarif(MOCK_FINDINGS)
    raw = json.dumps(sarif)
    parsed = json.loads(raw)
    assert parsed == sarif


def test_to_text_contains_summary():
    text = to_text(MOCK_FINDINGS)
    assert "LogNorm Report" in text
    assert "Semantic duplicates:" in text
    assert "Blind spots:" in text
    assert "Level mismatches:" in text


def test_to_text_lists_file_locations():
    text = to_text(MOCK_FINDINGS)
    assert "app.py:10" in text
    assert "payments.py:45" in text
    assert "handlers.py:30" in text


def test_sarif_end_to_end_three_rule_ids():
    """End-to-end: 3+ ruleIds, valid schema, correct structure."""
    sarif = to_sarif(MOCK_FINDINGS, tool_version="1.0.0")

    # Valid JSON round-trip
    raw = json.dumps(sarif, indent=2)
    reparsed = json.loads(raw)

    # Schema validation
    jsonschema.validate(instance=reparsed, schema=_SARIF_SCHEMA)

    results = reparsed["runs"][0]["results"]
    rule_ids = {r["ruleId"] for r in results}
    assert len(rule_ids) >= 3, f"Expected >=3 distinct ruleIds, got {rule_ids}"

    # Tool metadata
    assert reparsed["runs"][0]["tool"]["driver"]["name"] == "LogNorm"
    assert reparsed["runs"][0]["tool"]["driver"]["version"] == "1.0.0"

    # Every result has all required SARIF fields
    for r in results:
        assert "ruleId" in r
        assert "message" in r and "text" in r["message"]
        assert "locations" in r and len(r["locations"]) >= 1
        phys = r["locations"][0]["physicalLocation"]
        assert "artifactLocation" in phys and "uri" in phys["artifactLocation"]
        assert "region" in phys and "startLine" in phys["region"]
