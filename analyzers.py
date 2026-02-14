"""Core analysis engines for LogNorm."""
import ast
import hashlib
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


def detect_fingerprint_collisions(calls: List[LogCall]) -> List[List[LogCall]]:
    """Find identical log messages appearing in multiple locations."""
    from collections import defaultdict
    by_message: dict = defaultdict(list)
    for c in calls:
        if c.message and c.message != '<dynamic>':
            by_message[c.message].append(c)
    return [group for group in by_message.values() if len(group) > 1]


CONTEXT_IDENTIFIERS = {
    'request_id', 'req_id', 'trace_id', 'span_id',
    'user_id', 'uid', 'username',
    'order_id', 'transaction_id', 'tx_id',
    'session_id', 'correlation_id',
}


@dataclass
class MissingContext:
    log: LogCall
    available: List[str]


def detect_missing_context(
    source: str, filepath: str, calls: List[LogCall]
) -> List[MissingContext]:
    """Find log calls that omit important context variables available in scope."""
    tree = ast.parse(source)
    results: List[MissingContext] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scope_vars: set = set()
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg in CONTEXT_IDENTIFIERS:
                scope_vars.add(arg.arg)
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id in CONTEXT_IDENTIFIERS:
                        scope_vars.add(target.id)
        if not scope_vars:
            continue
        for call_node in ast.walk(node):
            if not (
                isinstance(call_node, ast.Call)
                and isinstance(call_node.func, ast.Attribute)
                and call_node.func.attr in LOG_METHODS
            ):
                continue
            call_names = {
                n.id for n in ast.walk(call_node) if isinstance(n, ast.Name)
            }
            missing = scope_vars - call_names
            if missing:
                matching = [
                    c for c in calls
                    if c.file == filepath and c.line == call_node.lineno
                ]
                if matching:
                    results.append(MissingContext(matching[0], sorted(missing)))
    return results
