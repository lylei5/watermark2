"""Paired spectral readouts; common-filter invariance is a hypothesis on real edits."""
import numpy as np

METHODS = ['shared_phase', 'shared_log_gain', 'content_phase', 'noise_phase',
           'phase_concat', 'permuted_phase', 'sscd', 'sscd_centered']


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def spectrum(x):
    x = np.asarray(x, dtype=np.float64).reshape(4, 64, 64)
    # Omit the zero horizontal frequency, using the same support in all arms.
    return np.fft.rfft2(x, axes=(-2, -1))[:, :, 1:]


def phase(x):
    return np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x) > 1e-12)


def pool(x, height=8, width=8):
    return x.reshape(4, height, 64 // height, width, 32 // width).mean(axis=(2, 4))


def complex_vector(x):
    return np.concatenate([x.real.ravel(), x.imag.ravel()]).astype(np.float32)


def describe(content, noise):
    a, b = spectrum(content), spectrum(noise)
    pa, pb = phase(a), phase(b)
    relative = pa * np.conj(pb)
    return dict(
        shared_phase=complex_vector(pool(relative)),
        shared_log_gain=pool(np.log(np.maximum(np.abs(a), 1e-12)) -
                             np.log(np.maximum(np.abs(b), 1e-12)), 16, 8).ravel().astype(np.float32),
        content_phase=complex_vector(pool(pa)),
        noise_phase=complex_vector(pool(pb)),
        phase_concat=np.concatenate([complex_vector(pool(pa, 8, 4)),
                                     complex_vector(pool(pb, 8, 4))]))


def mathematical_checks():
    """Numerical identities on arrays, not image attack generation."""
    rng = np.random.default_rng(2026111401)
    a, b = rng.normal(size=(2, 4, 64, 64))
    fy, fx = np.fft.fftfreq(64)[:, None], np.fft.rfftfreq(64)[None, :]
    h = np.exp(-4 * (fy ** 2 + fx ** 2)) * np.exp(-2j * np.pi * (2 * fy + 3 * fx))
    filtered = [np.fft.irfft2(np.fft.rfft2(x) * h, s=(64, 64)) for x in [a, b]]
    raw, common = describe(a, b), describe(*filtered)
    # Nyquist coefficients of an integer shift remain real, so this filter is real-valued.
    errors = {k: float(np.max(np.abs(raw[k] - common[k])))
              for k in ['shared_phase', 'shared_log_gain']}
    assert max(errors.values()) < 1e-6, errors
    assert all(v.shape == (512,) and np.isfinite(v).all() for v in raw.values())
    return errors
