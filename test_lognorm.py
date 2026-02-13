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
