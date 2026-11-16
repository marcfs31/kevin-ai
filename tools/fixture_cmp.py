"""Compare a regenerated fixture file with the one on disk.

The generators print float64 results of math.exp, math.cos and friends, and those are not
correctly rounded: macOS's libm and glibc can differ in the last bit, so a JSON fixture made on
one can differ from the same run on the other in the 16th or 17th digit. Bytes must match
exactly, except that in a .json file two numbers may differ by at most MAX_ULPS units in the
last place; the structure, every key, every string and every integer must still be identical.
"""

import json
import math

MAX_ULPS = 4


def _walk(a, b, where):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, int) and isinstance(b, int):
        return None if type(a) is type(b) and a == b else "%s: %r != %r" % (where, a, b)
    if isinstance(a, float) and isinstance(b, float):
        if a == b:
            return None
        if not (math.isfinite(a) and math.isfinite(b)):
            return "%s: %r != %r" % (where, a, b)
        tol = MAX_ULPS * math.ulp(max(abs(a), abs(b)))
        return None if abs(a - b) <= tol else "%s: %r != %r (more than %d ulps)" % (where, a, b, MAX_ULPS)
    if type(a) is not type(b):
        return "%s: %s != %s" % (where, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        if list(a) != list(b):
            return "%s: keys %s != %s" % (where, list(a)[:8], list(b)[:8])
        for k in a:
            msg = _walk(a[k], b[k], "%s.%s" % (where, k))
            if msg:
                return msg
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: length %d != %d" % (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            msg = _walk(x, y, "%s[%d]" % (where, i))
            if msg:
                return msg
        return None
    return None if a == b else "%s: %r != %r" % (where, a, b)


def diff(name, on_disk, regenerated):
    """None when the two files agree (see the module comment), else what differs first."""
    if on_disk == regenerated:
        return None
    if not name.endswith(".json"):
        return "%s: bytes differ" % name
    try:
        a, b = json.loads(on_disk), json.loads(regenerated)
    except ValueError as e:
        return "%s: not JSON: %s" % (name, e)
    msg = _walk(a, b, name)
    if msg:
        return msg
    # every value is within tolerance; if none actually differs, the bytes differ by formatting
    return "%s: same values, different bytes (formatting)" % name if a == b else None
