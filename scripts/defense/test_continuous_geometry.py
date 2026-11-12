"""Server-side mathematical and implementation checks; no image attacks."""
import argparse
import hashlib
import json
import secrets
import tempfile
from pathlib import Path
import numpy as np
from continuous_geometry_binding import ARMS, embed, decode, rotate, angles, uniforms
from anchor_binding import identity
from transport_binding import conditional_transport


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(202611130001)
    roundtrips = 0
    raw_errors = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for t in range(12):
            secret = {k: secrets.token_bytes(n).hex() for k, n in [('key', 32), ('nonce', 16), ('message', 32), ('binding_key', 32)]}
            secret['amplitude_seed'] = 202611130100 + t
            (root / f'{t:03}.json').write_text(json.dumps(secret))
            z = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
            logits = rng.standard_normal(1024)
            bundle = dict(code=(logits > 0).astype(np.uint8), logits=logits)
            _, cipher, _, key = identity(root, t)
            u = uniforms(key, b'check', 10000)
            assert np.all((u > 0) & (u < 1)) and np.array_equal(u, uniforms(key, b'check', 10000))
            assert not np.array_equal(u, uniforms(key, b'other', 10000))
            for arm in ARMS:
                y = embed(z, bundle, root, t, arm)
                assert y.dtype == np.float32 and np.isfinite(y).all()
                assert decode(y, bundle, root, t, arm) == 1
                roundtrips += 1
                if arm != 'full':
                    theta, perm = angles(bundle, root, t, arm)
                    back = rotate(rotate(z, theta, perm), theta, perm, inverse=True)
                    np.testing.assert_allclose(back, z, atol=2e-15, rtol=2e-15)
                    recovered = rotate(y, theta, perm, inverse=True).ravel()
                    raw_errors += int(np.sum((recovered > 0) != cipher))
                    np.testing.assert_allclose(np.sum(rotate(z, theta, perm)**2), np.sum(z.astype(float)**2), rtol=1e-14)
    assert raw_errors == 0
    n = 300000
    w = rng.standard_normal((n, 2))
    noise = rng.standard_normal(n)
    channel = []
    for sigma in [0., .4, 1.]:
        for delta in [0., .05, .1, .2, .5, 1.]:
            recovered = np.cos(delta) * w[:, 0] - np.sin(delta) * w[:, 1] + sigma * noise
            measured = float(np.mean((recovered > 0) == (w[:, 0] > 0)))
            exact = float(.5 + np.arcsin(np.cos(delta) / np.sqrt(1 + sigma**2)) / np.pi)
            se = np.sqrt(exact * (1 - exact) / n)
            assert abs(measured - exact) <= 5 * se + 1 / n
            channel.append(dict(sigma=sigma, angle=delta, n=n, measured=measured, analytic=exact, mc_standard_error=float(se)))
    smooth = []
    for sigma in [.4, 1.]:
        for delta in [.1, .01, .001]:
            rho = 1 / np.sqrt(1 + sigma**2)
            penalty = (np.arcsin(rho) - np.arcsin(rho * np.cos(delta))) / np.pi
            smooth.append(dict(sigma=sigma, delta=delta, penalty_over_delta_squared=float(penalty / delta**2), limit=float(1 / (2 * np.pi * sigma))))
    # Joint-Gaussian moment diagnostic with deliberately source-dependent angles.
    z = rng.standard_normal((n, 2))
    beta = rng.uniform(0, 2 * np.pi, n)
    theta = 3 * np.arctan2(z[:, 1], z[:, 0]) + beta
    c, s = np.cos(theta), np.sin(theta)
    u = np.column_stack((c * z[:, 0] + s * z[:, 1], -s * z[:, 0] + c * z[:, 1]))
    target = rng.integers(0, 2, (n, 2))
    encoded = conditional_transport(u, target)
    y = np.column_stack((c * encoded[:, 0] - s * encoded[:, 1], s * encoded[:, 0] + c * encoded[:, 1]))
    result = dict(passed=True, roundtrips=roundtrips, raw_sign_errors=raw_errors,
        orthogonal_inverse_and_energy_checks=True, finite_uniform_domain_separation_checks=True,
        channel=channel, small_angle=smooth,
        illustrative_distribution=dict(n=n, mean=y.mean(0).tolist(), covariance=np.cov(y.T).tolist(),
            cross_square_moment=float(np.mean(y[:, 0]**2 * y[:, 1]**2)), mse=float(np.mean((y-z)**2))),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['continuous_geometry_binding.py', 'transport_binding.py', 'test_continuous_geometry.py']},
        scope='Ideal Gaussian channel and mathematical roundtrip checks only; no image/model experiment or empirical authentication guarantee. Monte Carlo moments do not prove Gaussianity.')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['channel', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
