"""Tests for LogNorm analyzers."""
import pytest
from analyzers import (extract, detect_duplicates, detect_mismatches, LogCall,
                       generate_fingerprints, find_fingerprint_collisions)


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


# --- Fingerprint & collision tests ---


def test_generate_fingerprints_returns_correct_structure():
    code = '''
import logging
logger = logging.getLogger(__name__)

def foo():
    logger.info("hello world")

def bar():
    logger.info("goodbye world")
'''
    fps = generate_fingerprints(code, 'test.py')
    assert len(fps) == 2
    assert fps[0]['file'] == 'test.py'
    assert fps[0]['function_name'] == 'foo'
    assert fps[0]['raw_message'] == 'hello world'
    assert len(fps[0]['fingerprint']) == 12
    # Different messages -> different fingerprints
    assert fps[0]['fingerprint'] != fps[1]['fingerprint']


def test_collision_same_message_in_two_different_functions():
    """Identical hardcoded message in two functions = collision (can't trace in logs)."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def handle_request():
    logger.info("request handled")

def process_event():
    logger.info("request handled")
'''
    fps = generate_fingerprints(code, 'svc.py')
    collisions = find_fingerprint_collisions(fps)
    assert len(collisions) == 1
    assert len(collisions[0]) == 2
    funcs = {fp['function_name'] for fp in collisions[0]}
    assert funcs == {'handle_request', 'process_event'}


def test_no_collision_when_template_same_but_raw_different():
    """Messages that normalize to same template but differ in raw text are NOT collisions."""
    code_a = '''
import logging
logger = logging.getLogger(__name__)

def foo():
    logger.info("user %s connected")
'''
    code_b = '''
import logging
logger = logging.getLogger(__name__)

def bar():
    logger.info("user %d connected")
'''
    fps = generate_fingerprints(code_a, 'a.py') + generate_fingerprints(code_b, 'b.py')
    collisions = find_fingerprint_collisions(fps)
    assert len(collisions) == 0


def test_no_collision_single_occurrence():
    """A message appearing only once should never be reported as collision."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def foo():
    logger.info("unique message")
'''
    fps = generate_fingerprints(code, 'test.py')
    collisions = find_fingerprint_collisions(fps)
    assert len(collisions) == 0


def test_normalize_fstring_template():
    """f-string {expr} expressions should be replaced with {} in template."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def greet():
    name = "alice"
    logger.info(f"hello {name}, welcome to {name + \'!\'}")
'''
    fps = generate_fingerprints(code, 'test.py')
    assert fps[0]['raw_message'] == 'hello {}, welcome to {}'
    import hashlib
    expected = hashlib.sha256('test.py:greet:hello {}, welcome to {}'.encode()).hexdigest()[:12]
    assert fps[0]['fingerprint'] == expected


def test_normalize_percent_style_template():
    """%-style %s/%d/%f placeholders should be normalized to {} for fingerprinting."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def report():
    logger.info("processed %d items in %f seconds for user %s")
'''
    fps = generate_fingerprints(code, 'test.py')
    # raw_message preserves original placeholders
    assert fps[0]['raw_message'] == 'processed %d items in %f seconds for user %s'
    # fingerprint uses normalized template
    import hashlib
    normalized = 'processed {} items in {} seconds for user {}'
    expected = hashlib.sha256(f'test.py:report:{normalized}'.encode()).hexdigest()[:12]
    assert fps[0]['fingerprint'] == expected


def test_normalize_format_style_template():
    """.format()-style {name}/{0} placeholders should be normalized to {} for fingerprinting."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def notify():
    logger.info("user {name} completed {0} tasks".format(5, name="bob"))
'''
    fps = generate_fingerprints(code, 'test.py')
    # raw_message preserves original {name} and {0}
    assert fps[0]['raw_message'] == 'user {name} completed {0} tasks'
    # fingerprint normalizes them all to {}
    import hashlib
    normalized = 'user {} completed {} tasks'
    expected = hashlib.sha256(f'test.py:notify:{normalized}'.encode()).hexdigest()[:12]
    assert fps[0]['fingerprint'] == expected


def test_collision_across_different_files():
    """Same hardcoded message in different files should be detected."""
    code_a = '''
import logging
logger = logging.getLogger(__name__)

def handler_a():
    logger.warning("connection lost")
'''
    code_b = '''
import logging
logger = logging.getLogger(__name__)

def handler_b():
    logger.warning("connection lost")
'''
    fps = generate_fingerprints(code_a, 'a.py') + generate_fingerprints(code_b, 'b.py')
    collisions = find_fingerprint_collisions(fps)
    assert len(collisions) == 1
    files = {fp['file'] for fp in collisions[0]}
    assert files == {'a.py', 'b.py'}


def test_no_collision_same_message_same_function():
    """Same message twice in the same function is not a collision (same location)."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def retry_handler():
    logger.info("retrying")
    logger.info("retrying")
'''
    fps = generate_fingerprints(code, 'test.py')
    collisions = find_fingerprint_collisions(fps)
    assert len(collisions) == 0
