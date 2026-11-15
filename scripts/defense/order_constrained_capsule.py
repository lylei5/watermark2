"""Recover a registered random key from query-induced partial-order constraints.

This is an experimental decoder for the existing real-valued capsule format.
Ordinal invariance and isotonic projection are prior art; no security or novelty
claim follows from a successful decode. All numerical execution belongs on249.
"""
import hashlib
import hmac

import numpy as np
from scipy.optimize import isotonic_regression, minimize

import transported_key_capsule as capsule


def project_order(values, query):
    """Euclidean projection onto z_i <= z_j whenever query_i < query_j.

    Within a tied query group, sorting the current values adds no restriction
    at the optimum: an exchange argument preserves their original ordering.
    Tied query values therefore remain intervals, not forced equalities.
    """
    permutation = np.lexsort((values, query))
    fitted = isotonic_regression(values[permutation], increasing=True).x
    result = np.empty_like(values)
    result[permutation] = fitted
    return result


def order_residual(reference, query):
    residuals = [reference[:, c] - project_order(reference[:, c], query[:, c]) for c in range(3)]
    return float(np.sqrt(np.mean(np.square(residuals))))


def decode(record, image, provider):
    checked = {k: v for k, v in record.items() if k != 'mac'}
    expected = hmac.new(provider, capsule.canonical(checked), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(record['mac'], expected):
        return None, dict(reason='record_authentication')
    meta = record['metadata']
    h = capsule.unpack(meta['capsule']).reshape(-1, 3)
    y = capsule.rgb(image, 'order').reshape(-1, 3)
    basis, _ = capsule.basis(provider, meta['id'], 'spread')
    design = meta['alpha'] * basis.reshape(-1, capsule.BITS)
    calls = 0

    def objective(signs):
        nonlocal calls
        calls += 1
        source = h - (design @ signs)[:, None]
        residual = np.column_stack([source[:, c] - project_order(source[:, c], y[:, c]) for c in range(3)])
        # Tiny fixed numerical regularization selects the minimum-norm solution.
        regularization = 1e-8
        value = .5 * np.sum(residual ** 2) + .5 * regularization * (signs @ signs)
        gradient = -design.T @ residual.sum(axis=1) + regularization * signs
        return float(value), gradient

    result = minimize(objective, np.zeros(capsule.BITS), method='L-BFGS-B', jac=True,
                      bounds=[(-1., 1.)] * capsule.BITS,
                      options=dict(maxiter=250, ftol=1e-12, gtol=1e-7, maxls=30))
    signs = np.where(result.x >= 0, 1., -1.)
    key = np.packbits((signs > 0).astype(np.uint8)).tobytes()
    confirmed = hmac.compare_digest(meta['confirmation'], hashlib.sha256(b'transported/' + meta['id'].encode() + key).hexdigest())
    detail = dict(reason='key_recovered' if confirmed else 'key_confirmation_failure',
                  key_confirmed=confirmed, optimizer_success=bool(result.success),
                  optimizer_status=int(result.status), iterations=int(result.nit), objective_calls=calls,
                  relaxed_objective=float(result.fun),
                  decoded_order_residual=order_residual(h - (design @ signs)[:, None], y),
                  coefficient_error_rms=float(np.sqrt(np.mean((result.x - signs) ** 2))),
                  unique_query_levels=[int(len(np.unique(y[:, c]))) for c in range(3)])
    return (key if confirmed else None), detail


def recover(record, image, provider):
    key, detail = decode(record, image, provider)
    if key is None:
        return None, detail
    # Acceptance is exact registered-key confirmation and authenticated unwrap;
    # there is no image-residual threshold after the decode.
    payload = capsule.unwrap(record, key, provider, 0., 0.)
    return (key, payload), detail
