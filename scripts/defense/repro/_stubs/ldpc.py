"""Stub for the `ldpc` package, which only the PRC watermark needs.

The released PGID code imports PRCWatermark unconditionally at module scope, so
`import ldpc` is required even for a Gaussian-Shading-only run. No interpreter on
this host has both pandas/diffusers (semantic_forgery_py310) and ldpc
(pgid_py310), and the repo ships an ldpc wheel but no pandas wheel.

This shim is injected through PYTHONPATH. It satisfies the import and raises
loudly if the PRC decoder is ever actually used, so a GS-only run cannot silently
depend on a broken PRC path.
"""


class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError(
            'ldpc stub: the PRC watermark decoder is not available in this '
            'environment. GS-only runs do not need it; a PRC run does.'
        )


bp_decoder = _Unavailable
BpDecoder = _Unavailable


def __getattr__(name):
    # Import machinery probes dunder attributes (__path__, __all__, ...);
    # answering those with AttributeError keeps the module importable.
    if name.startswith('__'):
        raise AttributeError(name)
    raise RuntimeError('ldpc stub: %r is not available (PRC watermark only)' % name)
