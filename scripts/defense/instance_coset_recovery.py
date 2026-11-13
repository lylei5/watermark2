"""Instance-conditioned code-offset key recovery with authenticated records.

The native watermark is generated first and its existing verifier material is
wrapped at enrollment. Recovery takes a query feature, an authenticated helper
record and the provider secret; it receives no reference feature or source bits.
This is a computational key-binding prototype, not an entropy-certified FE.
"""
import base64
import hashlib
import hmac
import json
import secrets

import galois
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def pack(bits):
    return base64.b64encode(np.packbits(bits).tobytes()).decode()


def unpack(value, length):
    return np.unpackbits(np.frombuffer(base64.b64decode(value), dtype=np.uint8))[:length]


class InstanceRecovery:
    def __init__(self, projection):
        self.projection = np.asarray(projection, dtype=np.float32)
        assert self.projection.shape == (512, 8192)
        self.projection_hash = hashlib.sha256(self.projection.tobytes()).hexdigest()
        self.bch = galois.BCH(1023, d=255)
        assert self.bch.k == 133 and self.bch.t == 127

    def codeword(self, key):
        assert len(key) == 16
        message = np.concatenate([np.zeros(5, dtype=np.uint8),
                                  np.unpackbits(np.frombuffer(key, dtype=np.uint8))])
        return np.asarray(self.bch.encode(galois.GF2(message)), dtype=np.uint8)

    def enroll(self, record_id, reference, views, families, native_material, provider_secret,
               selector='orbit_margin', mode='coset', key=None):
        assert selector in ['clean_margin', 'orbit_margin', 'quotient_margin',
                            'tangent4_margin', 'tangent8_margin', 'tangent4_orbit']
        assert mode in ['fixed', 'templates', 'coset']
        assert len(views) == len(families) == 8 and len(set(families)) == 4
        key = secrets.token_bytes(16) if key is None else key
        transform = None
        if selector.startswith('tangent'):
            # An instance-specific edit quotient. Removing radial components keeps
            # the reference direction; only registered nuisance directions vanish.
            samples = (views if selector == 'tangent8_margin' else np.stack([
                views[np.array(families) == family].mean(axis=0)
                for family in sorted(set(families))]))
            tangent = samples - (samples @ reference)[:, None] * reference[None, :]
            _, singular, vh = np.linalg.svd(tangent, full_matrices=False)
            rank = int(np.sum(singular > max(float(singular[0])*1e-5, 1e-7)))
            stored = np.asarray(vh[:rank], dtype='<f2')
            transform = dict(kind='orthogonal_edit_quotient', rank=rank, dimensions=512,
                             dtype='<f2', basis=base64.b64encode(stored.tobytes()).decode())
            reference = self.transform_feature(reference, transform)
            views = self.transform_feature(views, transform)
        p0 = reference @ self.projection
        pv = views @ self.projection
        if selector in ['clean_margin', 'tangent4_margin', 'tangent8_margin']:
            reliability = np.abs(p0)
        elif selector in ['orbit_margin', 'tangent4_orbit']:
            # Registration-only edit variance; regularization is global and frozen.
            variance = np.mean((pv-p0)**2, axis=0)
            reliability = np.abs(p0)/np.sqrt(variance + .05*np.mean(variance) + 1e-12)
        else:
            # Preserve predictable between-state flips; penalize within-state noise.
            groups = [pv[np.array(families) == family] for family in sorted(set(families))]
            centers = np.stack([group.mean(axis=0) for group in groups])
            variance = np.mean(np.stack([np.mean((group-center)**2, axis=0)
                                        for group, center in zip(groups, centers)]), axis=0)
            margin = np.min(np.abs(np.concatenate([p0[None, :], centers])), axis=0)
            reliability = margin/np.sqrt(variance + .05*np.mean(variance) + 1e-12)
        chosen = np.sort(np.argsort(-reliability, kind='stable')[:1023])
        mask = np.zeros(8192, dtype=np.uint8)
        mask[chosen] = 1
        reference_bits = (p0[chosen] >= 0).astype(np.uint8)
        basis = []
        if mode != 'fixed':
            for family in sorted(set(families)):
                template = np.mean(pv[np.array(families) == family], axis=0)
                basis.append(((template[chosen] >= 0).astype(np.uint8) ^ reference_bits))
        word = self.codeword(key)
        metadata = dict(version=1, id=record_id, projection_sha256=self.projection_hash,
                        selector=selector, mode=mode, n=1023, k=133, t=127,
                        selected_mask=pack(mask), offset=pack(reference_bits ^ word),
                        edit_basis=[pack(x) for x in basis],
                        confirmation=hashlib.sha256(b'confirm/'+record_id.encode()+key).hexdigest())
        if transform is not None:
            metadata['feature_transform'] = transform
        aad = canonical(metadata)
        nonce = secrets.token_bytes(12)
        wrapping_key = hmac.new(provider_secret, b'wrap/'+record_id.encode()+key, hashlib.sha256).digest()
        record = dict(metadata=metadata, nonce=nonce.hex(),
                      wrapped=base64.b64encode(AESGCM(wrapping_key).encrypt(nonce, native_material, aad)).decode())
        record['mac'] = hmac.new(provider_secret, b'record/'+canonical(record), hashlib.sha256).hexdigest()
        return record, key

    @staticmethod
    def transform_feature(feature, transform):
        if transform is None:
            return feature
        assert transform['kind'] == 'orthogonal_edit_quotient' and transform['dimensions'] == 512
        rank = transform['rank']
        assert 0 <= rank <= 8 and transform['dtype'] == '<f2'
        stored = np.frombuffer(base64.b64decode(transform['basis']), dtype='<f2').reshape(rank, 512)
        # Re-orthogonalize after storage quantization; enrollment and recovery
        # use these identical serialized directions.
        basis = np.linalg.qr(stored.astype(np.float32).T, mode='reduced')[0]
        residual = feature - (feature @ basis) @ basis.T
        return residual / np.maximum(np.linalg.norm(residual, axis=-1, keepdims=True), 1e-12)

    def offsets(self, metadata):
        zero = np.zeros(1023, dtype=np.uint8)
        basis = [unpack(v, 1023) for v in metadata['edit_basis']]
        if metadata['mode'] == 'fixed':
            candidates = [zero]
        elif metadata['mode'] == 'templates':
            candidates = [zero, *basis]
        else:
            candidates = [zero]
            for row in basis:
                candidates += [old ^ row for old in candidates]
        seen = set()
        for candidate in candidates:
            token = np.packbits(candidate).tobytes()
            if token not in seen:
                seen.add(token)
                yield candidate

    def recover(self, record, query, provider_secret):
        authenticated = {k:v for k,v in record.items() if k != 'mac'}
        expected_mac = hmac.new(provider_secret, b'record/'+canonical(authenticated), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(record['mac'], expected_mac):
            return None, dict(reason='record_authentication', attempts=0)
        metadata = record['metadata']
        assert metadata['projection_sha256'] == self.projection_hash
        assert (metadata['n'], metadata['k'], metadata['t']) == (1023, 133, 127)
        chosen = np.flatnonzero(unpack(metadata['selected_mask'], 8192))
        assert len(chosen) == 1023
        query = self.transform_feature(query, metadata.get('feature_transform'))
        observed = ((query @ self.projection[:, chosen]) >= 0).astype(np.uint8)
        base = observed ^ unpack(metadata['offset'], 1023)
        attempts = 0
        for edit in self.offsets(metadata):
            attempts += 1
            message, errors = self.bch.decode(galois.GF2(base ^ edit), errors=True)
            if int(errors) < 0 or np.any(np.asarray(message[:5])):
                continue
            key = np.packbits(np.asarray(message[5:], dtype=np.uint8)).tobytes()
            check = hashlib.sha256(b'confirm/'+metadata['id'].encode()+key).hexdigest()
            if not hmac.compare_digest(check, metadata['confirmation']):
                continue
            # Re-encode to enforce the exact bounded-distance recovery region.
            word = self.codeword(key)
            residual = int(np.count_nonzero(base ^ edit ^ word))
            if residual > 127:
                continue
            wrapping_key = hmac.new(provider_secret, b'wrap/'+metadata['id'].encode()+key, hashlib.sha256).digest()
            material = AESGCM(wrapping_key).decrypt(bytes.fromhex(record['nonce']),
                base64.b64decode(record['wrapped']), canonical(metadata))
            residual = min(int(np.count_nonzero(base ^ other ^ word)) for other in self.offsets(metadata))
            return (key, material), dict(reason='recovered', attempts=attempts, residual=residual)
        return None, dict(reason='no_confirmed_key', attempts=attempts)

    def recovery_distance(self, record, query, reference):
        """Reference-informed comparison control, never called by recover()."""
        metadata = record['metadata']
        chosen = np.flatnonzero(unpack(metadata['selected_mask'], 8192))
        query = self.transform_feature(query, metadata.get('feature_transform'))
        reference = self.transform_feature(reference, metadata.get('feature_transform'))
        transform = self.projection[:, chosen]
        delta = ((query @ transform) >= 0) ^ ((reference @ transform) >= 0)
        return min(int(np.count_nonzero(delta ^ shift)) for shift in self.offsets(metadata))
