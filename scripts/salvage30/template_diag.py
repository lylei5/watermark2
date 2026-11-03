"""Offline diagnostic (evaluation-only, keys available): is the reference sign template reliable?

For each of the 30 pairs we decrypt the reference and the forged candidate with the
private GS material and compare:
  - the true per-bit margin M_k (after ChaCha20 de-whitening, 64 votes),
  - the key-free template margin M_hat_k = sum_i sign(z_ref,i) sign(z_cand,i).
Also measures the reference's per-coordinate sign error rate against the true message.
This uses the keys for diagnosis only; it is never an attack input.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/confirmation100_baseline_20261102T204500Z'
PRIV = ROOT / 'private_eval/confirmation100_baseline_20261102T204500Z'

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms  # noqa: E402


def stream_bits(raw, n=16384):
    cipher = Cipher(algorithms.ChaCha20(raw[32:64], raw[64:]), mode=None).decryptor()
    ks = cipher.update(b'\x00' * (n // 8)) + cipher.finalize()
    return np.unpackbits(np.frombuffer(ks, dtype=np.uint8), bitorder='big')


def decrypt_bits(sign_bits, raw):
    cipher = Cipher(algorithms.ChaCha20(raw[32:64], raw[64:]), mode=None).decryptor()
    plain = cipher.update(np.packbits(sign_bits, bitorder='big').tobytes()) + cipher.finalize()
    return np.unpackbits(np.frombuffer(plain, dtype=np.uint8), bitorder='big')


def main():
    rows = []
    msg_errors = []
    for sid in range(1, 31):
        raw = (PRIV / (str(sid) + '.bin')).read_bytes()
        message = np.unpackbits(np.frombuffer(raw[:32], dtype=np.uint8), bitorder='big')
        z_ref = np.load(RUN / 'evaluate' / (str(sid) + '_reference') / 'target_inverse.npy').reshape(-1)
        z_c = np.load(RUN / 'evaluate' / (str(sid) + '_30') / 'target_inverse.npy').reshape(-1)
        ref_bits = (z_ref >= 0).astype(np.uint8)
        cand_bits = (z_c >= 0).astype(np.uint8)
        p = stream_bits(raw)
        d_ref = decrypt_bits(ref_bits, raw).reshape(64, 256)
        d_c = decrypt_bits(cand_bits, raw).reshape(64, 256)
        # true margins after de-whitening
        m_true = (2 * d_c.sum(axis=0).astype(np.int64) - 64)
        m_hat = ((2 * ref_bits.astype(np.int64) - 1) * (2 * cand_bits.astype(np.int64) - 1)).reshape(64, 256).sum(axis=0)
        ref_ba = float(np.mean((d_ref.sum(axis=0) > 32) == message))
        # reference per-coordinate sign errors against the true encoded bit
        expected = ((message[None, :] ^ p.reshape(64, 256)) > 0).astype(np.uint8)
        err = int((ref_bits.reshape(64, 256) != expected).sum())
        signs_agree = float(np.mean(np.sign(m_hat) == np.sign(m_true)))
        rho, _ = spearmanr(m_hat, m_true)
        rows.append(dict(task_id=sid, ref_ba=ref_ba, ref_coord_error=err / 16384.0,
                         m_hat_sign_agree=signs_agree, spearman=float(rho),
                         mean_abs_mhat=float(np.mean(np.abs(m_hat)))))
        msg_errors.append(err / 16384.0)
    print(json.dumps(dict(n=len(rows),
                          ref_ba_mean=float(np.mean([r['ref_ba'] for r in rows])),
                          ref_coord_error_mean=float(np.mean(msg_errors)),
                          ref_coord_error_max=float(np.max(msg_errors)),
                          m_hat_sign_agree_mean=float(np.mean([r['m_hat_sign_agree'] for r in rows])),
                          spearman_mean=float(np.mean([r['spearman'] for r in rows])),
                          rows=rows), indent=2))


if __name__ == '__main__':
    main()

