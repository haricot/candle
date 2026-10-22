"""Strict f32 byte/ULP audit; numerical gates never substitute for exact gates."""
from __future__ import annotations

import numpy as np


def tensor_metrics(reference, candidate, *, atol=1e-3, min_cosine=.999):
    a = np.ascontiguousarray(reference, dtype='<f4')
    b = np.ascontiguousarray(candidate, dtype='<f4')
    if a.shape != b.shape:
        return {'bitwise_equal': False, 'numeric_pass': False, 'status': 'SHAPE_MISMATCH',
                'reference_shape': list(a.shape), 'candidate_shape': list(b.shape)}
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('R7 refuses empty/nonfinite tensors')
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    different = ua != ub
    xor = np.bitwise_xor(ua, ub)
    # Count actual changed bits, separately from changed f32 words.
    bit_count = int(np.unpackbits(xor.view(np.uint8)).sum(dtype=np.uint64))
    def ordered(u):
        return np.where(u & np.uint32(0x80000000), ~u, u ^ np.uint32(0x80000000)).astype(np.int64)
    ulp = np.abs(ordered(ua) - ordered(ub))
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = np.abs(x-y)
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    cosine = float(np.sum(x*y)/norm) if norm else (1. if not delta.any() else 0.)
    first = None
    if different.any():
        index = tuple(int(i) for i in np.argwhere(different)[0])
        first = {'index': list(index), 'reference': float(a[index]), 'candidate': float(b[index]),
                 'reference_bits': f'{int(ua[index]):08x}', 'candidate_bits': f'{int(ub[index]):08x}',
                 'ulp_distance': int(ulp[index])}
    exact = not different.any()
    return {'shape': list(a.shape), 'elements': int(a.size), 'bitwise_equal': bool(exact),
            'status': 'PASS_BITWISE' if exact else 'DIFFERENT_F32_BITS',
            'mismatching_f32_words': int(different.sum()), 'mismatching_bits': bit_count,
            'max_ulp': int(ulp.max()), 'first_difference': first,
            'max_abs': float(delta.max()), 'rmse': float(np.sqrt(np.mean((x-y)**2))),
            'cosine': cosine, 'numeric_pass': bool(delta.max() <= atol and cosine >= min_cosine),
            'numeric_limits': {'max_abs': atol, 'min_cosine': min_cosine}}


def compare_decisions(reference, candidate):
    keys = ('frame', 'token', 'duration_index', 'duration', 'advance', 'forced_advance')
    first = None
    for i, (a, b) in enumerate(zip(reference, candidate)):
        differences = {key: {'reference': a[key], 'candidate': b[key]} for key in keys if a[key] != b[key]}
        if differences:
            first = {'decision': i, 'differences': differences}
            break
    if first is None and len(reference) != len(candidate):
        first = {'decision': min(len(reference), len(candidate)), 'reason': 'DIFFERENT_DECISION_COUNTS'}
    return {'exact': first is None, 'reference_decisions': len(reference),
            'candidate_decisions': len(candidate), 'first_difference': first}


def advance(frame, token, blank, duration, symbols, maximum):
    """Oracle for the production NeMo batched TDT advancement contract."""
    if min(frame, token, blank, duration, symbols) < 0 or (maximum is not None and maximum <= 0):
        raise ValueError('invalid TDT transition')
    count = symbols + (token != blank)
    effective = max(duration, 1) if token == blank else duration
    forced = effective == 0 and maximum is not None and count >= maximum
    return frame + effective + int(forced), (0 if effective or forced else count), bool(forced)
