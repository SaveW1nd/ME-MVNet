from __future__ import annotations

import numpy as np

from src.data.isrj_generator import add_awgn_by_jnr, make_baseband_lfm, make_single_isrj


def test_generator_tf_relation_and_mask_segments() -> None:
    fs = 40e6
    tp = 100e-6
    b = 10e6
    kr = b / tp
    n = int(round(fs * tp))

    s = make_baseband_lfm(fs=fs, tp=tp, kr=kr)
    nl = 200
    nf = 2
    j, mask = make_single_isrj(s=s, nl=nl, nf=nf)

    assert s.shape == (n,)
    assert j.shape == (n,)
    assert mask.shape == (n,)
    assert mask.dtype == np.uint8

    nu = (nf + 1) * nl
    # first cycle: [0:nl] sample segment, then two forwarding segments
    assert np.all(mask[0:nl] == 0)
    assert np.all(mask[nl : 2 * nl] == 1)
    assert np.all(mask[2 * nl : 3 * nl] == 1)
    assert np.all(mask[3 * nl : nu] == 0)

    # j is only non-zero inside forwarding segments
    assert np.all(np.abs(j[mask == 0]) == 0)
    assert np.all(np.abs(j[mask == 1]) > 0)

    tl = nl / fs
    tf = (nf * nl) / fs
    assert abs(tf - nf * tl) < 1e-12


def test_awgn_by_jnr_shape_and_dtype() -> None:
    rng = np.random.default_rng(123)
    j = np.ones((4000,), dtype=np.complex64)
    x = add_awgn_by_jnr(j=j, jnr_db=0.0, rng=rng)
    assert x.shape == j.shape
    assert x.dtype == np.complex64
    assert np.mean(np.abs(x - j)) > 0
