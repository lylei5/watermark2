"""Forward-transported real-valued key capsule; research prototype, CPU249."""
import base64
import hashlib
import hmac
import json
import secrets

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from scipy.optimize import nnls

SIZE = 64
BITS = 128
SIGMAS = [0., .5, 1., 2.]
RADIUS = 6
_basis_cache = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def pack(x):
    x = np.asarray(x, dtype='<f4')
    return dict(shape=list(x.shape), data=base64.b64encode(x.tobytes()).decode())


def unpack(x):
    return np.frombuffer(base64.b64decode(x['data']), dtype='<f4').reshape(x['shape']).astype(np.float64)


def rgb(image, observation='area'):
    x = np.asarray(image.convert('RGB'), dtype=np.float64) / 255.
    if observation == 'order':
        assert x.shape == (512, 512, 3)
        blocks = x.reshape(SIZE, 8, SIZE, 8, 3).transpose(0, 2, 1, 3, 4).reshape(SIZE, SIZE, 64, 3)
        # A single order statistic commutes with monotone clipping; an average does not.
        return np.partition(blocks, 31, axis=2)[:, :, 31, :]
    assert observation == 'area'
    return cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def masks():
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cells = (xx // 16 + yy // 16) % 2
    pilot = (cells == 0) & (xx % 16 >= 6) & (xx % 16 <= 9) & (yy % 16 >= 6) & (yy % 16 <= 9)
    witness = cells == 1
    decode = witness & ((xx + yy) % 2 == 0)
    check = witness & ~decode
    return pilot, decode, check, witness


def blur(x, sigma):
    return x.copy() if sigma == 0 else cv2.GaussianBlur(x, (13, 13), sigma, borderType=cv2.BORDER_REFLECT_101)


def bank(x):
    return np.stack([blur(x, s) for s in SIGMAS], axis=-1)


def basis(provider, identity, kind='dct'):
    suffix = b'' if kind == 'dct' else b'/' + kind.encode()
    token = hmac.new(provider, b'capsule-basis/' + identity.encode() + suffix, hashlib.sha256).digest()
    cache_key = token.hex()
    if cache_key not in _basis_cache:
        rng = np.random.default_rng(int.from_bytes(token[:16], 'big'))
        if kind == 'dct':
            coords = (np.arange(SIZE) + .5) / SIZE
            modes = sorted([(i, j) for i in range(16) for j in range(16) if i + j], key=lambda z: (z[0]**2 + z[1]**2, z))[:BITS]
            raw = np.stack([np.cos(np.pi * i * coords)[None, :] * np.cos(np.pi * j * coords)[:, None] for i, j in modes], axis=-1)
        else:
            assert kind == 'spread'
            white = rng.normal(size=(SIZE, SIZE, BITS))
            raw = np.stack([cv2.GaussianBlur(white[..., i], (5, 5), .75, borderType=cv2.BORDER_REFLECT_101) for i in range(BITS)], axis=-1)
        raw *= masks()[3][..., None]
        q, _ = np.linalg.qr(raw.reshape(-1, BITS))
        rotation, _ = np.linalg.qr(rng.normal(size=(BITS, BITS)))
        b = (q @ rotation).reshape(SIZE, SIZE, BITS) * np.sqrt(SIZE * SIZE / BITS)
        # Gaussian filtering is spatial only; basis columns remain separate.
        bb = np.stack([np.stack([blur(b[..., i], s) for i in range(BITS)], axis=-1) for s in SIGMAS], axis=-1)
        assert np.max(np.abs(bb[masks()[0]])) < 1e-12
        _basis_cache[cache_key] = b, bb
    return _basis_cache[cache_key]


def enroll(identity, image, payload, provider, alpha, key=None, mode='transport'):
    key = secrets.token_bytes(16) if key is None else key
    signs = np.unpackbits(np.frombuffer(key, dtype=np.uint8)).astype(np.float64) * 2 - 1
    kind = 'spread' if mode.startswith('spread_') else 'dct'
    observation = 'order' if '_order' in mode else 'area'
    b, _ = basis(provider, identity, kind)
    capsule = rgb(image, observation) + alpha * (b @ signs)[..., None]
    meta = dict(version=1, id=identity, alpha=alpha, capsule=pack(capsule),
                confirmation=hashlib.sha256(b'transported/' + identity.encode() + key).hexdigest())
    if mode not in ['transport', 'identity']:
        meta.update(version=2, basis_kind=kind, observation=observation)
    nonce = secrets.token_bytes(12)
    wrapping = hmac.new(provider, b'transported-wrap/' + identity.encode() + key, hashlib.sha256).digest()
    record = dict(metadata=meta, nonce=nonce.hex(), wrapped=base64.b64encode(AESGCM(wrapping).encrypt(nonce, payload, canonical(meta))).decode())
    record['mac'] = hmac.new(provider, canonical(record), hashlib.sha256).hexdigest()
    return record, key


def observe(record, image, provider, mode='transport'):
    checked = {k: v for k, v in record.items() if k != 'mac'}
    if not hmac.compare_digest(record['mac'], hmac.new(provider, canonical(checked), hashlib.sha256).hexdigest()):
        return None, dict(reason='record_authentication')
    meta = record['metadata']; h = unpack(meta['capsule']); y = rgb(image, meta.get('observation', 'area'))
    b, bb = basis(provider, meta['id'], meta.get('basis_kind', 'dct')); hb = bank(h)
    pilot, decode, check, _ = masks()
    usable = (y > .02) & (y < .98)
    weights = []; offsets = []; valid_counts = []; active = []
    partial = mode in ['spread_area', 'spread_order', 'dct_order']
    for c in range(3):
        valid = pilot & usable[..., c]; valid_counts.append(int(valid.sum()))
        if mode == 'identity' or mode.endswith('_identity'):
            w = np.array([1., 0., 0., 0.]); offset = 0.
        else:
            if valid.sum() < 24:
                if partial:
                    weights.append(np.zeros(4)); offsets.append(0.); active.append(False); continue
                return None, dict(reason='insufficient_pilots', pilot_counts=valid_counts)
            a = hb[..., c, :][valid]; target = y[..., c][valid]
            ac = a - a.mean(axis=0); yc = target - target.mean()
            # Small identity-centered regularization stabilizes collinear blur columns.
            reg = .001
            w, _ = nnls(np.vstack([ac, np.eye(4) * reg]), np.r_[yc, reg, 0., 0., 0.], maxiter=100)
            offset = float(target.mean() - a.mean(axis=0) @ w)
            if not .05 <= w.sum() <= 12:
                if partial:
                    weights.append(np.zeros(4)); offsets.append(0.); active.append(False); continue
                return None, dict(reason='unidentifiable_channel', pilot_counts=valid_counts)
        weights.append(w); offsets.append(offset); active.append(True)
    designs = []; values = []; held_designs = []; held_values = []
    for c in range(3):
        transported = meta['alpha'] * (bb @ weights[c])
        response = hb[..., c, :] @ weights[c] + offsets[c] - y[..., c]
        d = decode & usable[..., c] & active[c]; v = check & usable[..., c] & active[c]
        designs.append(transported[d]); values.append(response[d])
        held_designs.append(transported[v]); held_values.append(response[v])
    a = np.concatenate(designs); v = np.concatenate(values)
    av = np.concatenate(held_designs); vv = np.concatenate(held_values)
    detail = dict(reason='decoding', pilot_counts=valid_counts, decode_coordinates=len(v), check_coordinates=len(vv),
                  weights=[w.tolist() for w in weights], offsets=offsets, active_channels=active)
    if len(v) < 256 or len(vv) < 256:
        return None, dict(detail, reason='insufficient_coordinates')
    gram = a.T @ a
    eigen = np.linalg.eigvalsh(gram)
    detail.update(min_eigenvalue=float(eigen[0]), max_eigenvalue=float(eigen[-1]))
    if eigen[0] <= max(1e-10, eigen[-1] * 1e-9):
        return None, dict(detail, reason='rank_failure')
    estimate = np.linalg.solve(gram, a.T @ v)
    key = np.packbits((estimate >= 0).astype(np.uint8)).tobytes()
    signs = (estimate >= 0).astype(np.float64) * 2 - 1
    residual = float(np.sqrt(np.mean(np.square(av @ signs - vv))))
    detail.update(residual=residual, coefficient_error_rms=float(np.sqrt(np.mean(np.square(estimate - signs)))),
                  key_confirmed=hmac.compare_digest(meta['confirmation'], hashlib.sha256(b'transported/' + meta['id'].encode() + key).hexdigest()))
    if not detail['key_confirmed']:
        return None, dict(detail, reason='key_confirmation_failure')
    return key, dict(detail, reason='key_recovered')


def reference_scores(reference, image, detail, observation='area'):
    x = rgb(reference, observation); y = rgb(image, observation); xb = bank(x)
    check = masks()[2]
    raw = []; fitted = []
    for c in range(3):
        valid = check & (y[..., c] > .02) & (y[..., c] < .98)
        raw.extend((x[..., c][valid] - y[..., c][valid]).tolist())
        if 'weights' in detail and detail.get('active_channels', [True]*3)[c]:
            pred = xb[..., c, :] @ np.asarray(detail['weights'][c]) + detail['offsets'][c]
            fitted.extend((pred[valid] - y[..., c][valid]).tolist())
    return dict(pixel_residual=float(np.sqrt(np.mean(np.square(raw)))) if raw else None,
                reference_residual=float(np.sqrt(np.mean(np.square(fitted)))) if fitted else None)


def unwrap(record, key, provider, residual, threshold):
    if threshold is None or residual > threshold:
        return None
    meta = record['metadata']
    if not hmac.compare_digest(meta['confirmation'], hashlib.sha256(b'transported/' + meta['id'].encode() + key).hexdigest()):
        return None
    wrapping = hmac.new(provider, b'transported-wrap/' + meta['id'].encode() + key, hashlib.sha256).digest()
    return AESGCM(wrapping).decrypt(bytes.fromhex(record['nonce']), base64.b64decode(record['wrapped']), canonical(meta))
