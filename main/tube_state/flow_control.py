"""Analytic predicted-terminal correction in the existing normalized block basis."""
from __future__ import annotations
import numpy as np
from . import projection_margin as carrier


def projection_record(z, book, message):
    x = carrier.blocks(np.asarray(z)).reshape(carrier.SUPPORT_COUNT, -1).astype(np.float64)
    d = book['directions'].astype(np.float64)
    p = np.einsum('ij,ij->i', x, d)
    c = book['codes'][message]
    deficit = np.maximum(0., 1. - c * p)
    return {'projection': p.tolist(), 'signed_projection': (c*p).tolist(),
            'loss': float(np.mean(deficit**2))}


def request(z0hat, book, message):
    """Return u, minus a preconditioned analytic loss gradient; no autograd."""
    record = projection_record(z0hat, book, message)
    d = book['directions'].astype(np.float64)
    c = book['codes'][message]
    deficit = np.maximum(0., 1. - c * np.asarray(record['projection']))
    values = (deficit*c/3./np.einsum('ij,ij->i', d, d))[:, None]*d
    u = np.zeros_like(z0hat, dtype=np.float32)
    carrier.put_blocks(u, values.astype(np.float32))
    return u, record


def rms(value, support=True):
    x = np.asarray(value, dtype=np.float64)
    if support:
        x = carrier.blocks(x)
    return float(np.sqrt(np.mean(x*x)))
