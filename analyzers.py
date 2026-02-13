"""Core analysis engines for LogNorm."""
import ast
import hashlib
import re
from dataclasses import dataclass

from typing import List, Tuple

LOG_METHODS = {
    'debug': 1, 'info': 2, 'warning': 3, 'warn': 3,
    'error': 4, 'critical': 5, 'exception': 4,
}
SEVERE_WORDS = {
    'fail', 'error', 'crash', 'timeout', 'fatal',
    'exception', 'refused', 'denied', 'abort', 'panic',
}
MILD_WORDS = {
    'success', 'complete', 'loaded', 'updated',
    'created', 'started', 'ready', 'done',
}


@dataclass
class LogCall:
    file: str
    line: int
    func: str
    level: str
    message: str
    fingerprint: str


@dataclass
class BlindSpot:
    file: str
    line: int
    kind: str


@dataclass
class Mismatch:
    log: LogCall
    expected_min: str
    reason: str


def _extract_msg(node):
    if not node.args:
        return ''
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        return ''.join(
            str(v.value) if isinstance(v, ast.Constant) else '{}'
            for v in arg.values
        )
    return '<dynamic>'


class LogExtractor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.calls: List[LogCall] = []
        self.blindspots: List[BlindSpot] = []
        self._func_stack = ['<module>']

    def visit_FunctionDef(self, node):
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS:
            msg = _extract_msg(node)
            func = self._func_stack[-1]
            raw = f"{self.filepath}:{func}:{msg}"
            fp = hashlib.md5(raw.encode()).hexdigest()[:12]
            self.calls.append(LogCall(
                self.filepath, node.lineno, func, node.func.attr, msg, fp
            ))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        has_log = any(
            isinstance(n, ast.Call)
            and isinstance(getattr(n, 'func', None), ast.Attribute)
            and getattr(n.func, 'attr', '') in LOG_METHODS
            for n in ast.walk(node)
        )
        if not has_log:
            self.blindspots.append(
                BlindSpot(self.filepath, node.lineno, 'silent_except')
            )
        self.generic_visit(node)


def extract(source: str, filepath: str) -> Tuple[List[LogCall], List[BlindSpot]]:
    tree = ast.parse(source)
    v = LogExtractor(filepath)
    v.visit(tree)
    return v.calls, v.blindspots


def detect_duplicates(calls: List[LogCall], threshold: float = 0.8) -> List[List[LogCall]]:
    filtered = [c for c in calls if c.message and c.message != '<dynamic>']
    if len(filtered) < 2:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vecs = TfidfVectorizer().fit_transform([c.message for c in filtered])
    sim = cosine_similarity(vecs)
    visited, groups = set(), []
    for i in range(len(filtered)):
        if i in visited:
            continue
        group = [filtered[i]]
        for j in range(i + 1, len(filtered)):
            if j not in visited and sim[i, j] >= threshold:
                group.append(filtered[j])
                visited.add(j)
        if len(group) > 1:
            groups.append(group)
            visited.add(i)
    return groups


def detect_mismatches(calls: List[LogCall]) -> List[Mismatch]:
    results = []
    for c in calls:
        if not c.message or c.message == '<dynamic>':
            continue
        low = c.message.lower()
        has_severe = any(w in low for w in SEVERE_WORDS)
        has_mild = any(w in low for w in MILD_WORDS)
        level_num = LOG_METHODS.get(c.level, 2)
        if has_severe and level_num < 3:
            results.append(Mismatch(
                c, 'warning', f'severe words in message but logged at {c.level}'
            ))
        if has_mild and not has_severe and level_num >= 4:
            results.append(Mismatch(
                c, 'info', f'mild message logged at {c.level}'
            ))
    return results


def _normalize_message_template(message: str) -> str:
    """Normalize log message template by replacing all variable placeholders with {}."""
    # Replace %-style: %s, %d, %f
    result = re.sub(r'%[sdf]', '{}', message)
    # Replace .format()-style named/positional and f-string expressions: {name}, {0}, {expr}
    result = re.sub(r'\{[^}]+\}', '{}', result)
    return result


def generate_fingerprints(source_code: str, filename: str) -> list:
    """Generate structured fingerprints for each log statement in source code.

    Returns list of dicts: {file, function_name, line, raw_message, fingerprint}.
    Fingerprint = SHA-256[:12] of 'filename:function_name:normalized_message'.
    """
    calls, _ = extract(source_code, filename)
    fingerprints = []
    for call in calls:
        normalized = _normalize_message_template(call.message)
        key = f"{filename}:{call.func}:{normalized}"
        fp = hashlib.sha256(key.encode()).hexdigest()[:12]
        fingerprints.append({
            'file': filename,
            'function_name': call.func,
            'line': call.line,
            'raw_message': call.message,
            'fingerprint': fp,
        })
    return fingerprints


def find_fingerprint_collisions(fingerprints: list) -> list:
    """Find groups where identical raw_message appears in different file:function locations.

    Returns list of collision groups (each group is a list of fingerprint dicts).
    Only reports messages that appear in >1 distinct file:function pair.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for fp in fingerprints:
        groups[fp['raw_message']].append(fp)

    collisions = []
    for raw_msg, items in groups.items():
        locations = {(item['file'], item['function_name']) for item in items}
        if len(locations) > 1:
            collisions.append(items)

    return collisions
