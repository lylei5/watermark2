"""Stub for the `galois` package, which only the PRC watermark needs.

`watermarks/prc_watermark.py` evaluates `GF = galois.GF(2)` at module scope, so the
name must exist for the module to import even in a GS-only run. Nothing else in a
GS-only path touches the field, so a placeholder class is enough; instantiating it
raises so a PRC run cannot silently proceed.
"""


class _UnavailableField:
    def __init__(self, *a, **k):
        raise RuntimeError(
            'galois stub: Galois fields are only needed by the PRC watermark, '
            'which is not installed in this environment.'
        )


def GF(order=None, *a, **k):  # noqa: N802 - mirrors the upstream API name
    return _UnavailableField


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    raise RuntimeError('galois stub: %r is not available (PRC watermark only)' % name)
