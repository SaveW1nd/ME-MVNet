import json
from pathlib import Path
import numpy as np


# =========================
# 0) 固定：体制/参数（来自文献）
# =========================
FS = 40e6        # 40 MHz  (Remote Sensing 2025 Table 1)  :contentReference[oaicite:9]{index=9}
TP = 100e-6      # 100 us  (Remote Sensing 2025 Table 1)  :contentReference[oaicite:10]{index=10}
B  = 10e6        # 10 MHz  (Remote Sensing 2025 Table 1)  :contentReference[oaicite:11]{index=11}
KR = B / TP      # chirp rate

N = int(round(FS * TP))  # 4000

# JNR = 10..25 dB  (Sensors 2021 uses 10~25 dB Monte Carlo) :contentReference[oaicite:12]{index=12}
JNR_DB_LIST = np.arange(10, 26, 1, dtype=np.int32)  # 10..25 inclusive

# forwarding times N_F in {1,2,4} (from open ISRJ dataset script repetition_times=[4,2,1]) :contentReference[oaicite:13]{index=13}
NF_LIST = np.array([1, 2, 4], dtype=np.int32)

# Tl ranges (Remote Sensing 2025 Table 1 gives ISPRJ 5–25, with TP=100us) :contentReference[oaicite:14]{index=14}
# Discretize by sample counts Nl = Tl * Fs
NL_MIN = int(round(5e-6  * FS))   # 200
NL_MAX_12 = int(round(25e-6 * FS))  # 1000  (for NF=1,2)
NL_MAX_4  = int(round(20e-6 * FS))  # 800   (for NF=4, because (NF+1)*Tl <= TP)

# 25 discrete Tl points each NF (fixed, deterministic)
NL_LIST_NF12 = np.unique(np.round(np.linspace(NL_MIN, NL_MAX_12, 25)).astype(np.int32))
NL_LIST_NF4  = np.unique(np.round(np.linspace(NL_MIN, NL_MAX_4,  25)).astype(np.int32))
assert len(NL_LIST_NF12) == 25
assert len(NL_LIST_NF4)  == 25

MC_REPEATS = 5            # 5 Monte-Carlo noise realizations per (NF, Nl, JNR) tuple
SEED = 20260303           # fixed seed for reproducibility
OUT_DIR = Path("isrj_single_dataset_v1")


# =========================
# 1) 基带 LFM（单脉冲）
# =========================
def make_baseband_lfm(fs: float, tp: float, kr: float) -> np.ndarray:
    """Baseband LFM chirp: exp(j*pi*kr*t^2), t centered in [-tp/2, tp/2)."""
    n = int(round(fs * tp))
    t = (np.arange(n) - n / 2) / fs
    s = np.exp(1j * np.pi * kr * t**2).astype(np.complex64)
    return s


# =========================
# 2) 单一 ISRJ 生成（机理实现）
# =========================
def make_single_isrj(s: np.ndarray, nl: int, nf: int) -> tuple[np.ndarray, np.ndarray]:
    """
    s: transmitted LFM (complex, length N)
    nl: slice length in samples  (Nl = Tl*Fs)
    nf: forwarding times         (N_F)
    Returns:
      j: ISRJ-only waveform (complex, length N)
      mask: forwarding mask (uint8, length N), 1 indicates forwarding segments
    """
    n = s.shape[0]
    nu = (nf + 1) * nl  # interrupted-sampling interval in samples
    k_cycles = n // nu  # full cycles

    j = np.zeros(n, dtype=np.complex64)
    mask = np.zeros(n, dtype=np.uint8)

    for k in range(k_cycles):
        base = k * nu
        sl = s[base: base + nl]  # sampled slice (not transmitted)

        # forward m=1..nf
        for m in range(1, nf + 1):
            st = base + m * nl
            ed = st + nl
            j[st:ed] += sl
            mask[st:ed] = 1

    return j, mask


# =========================
# 3) 加噪：按 JNR 定义生成 AWGN
# =========================
def add_awgn_by_jnr(j: np.ndarray, jnr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    JNR = 10*log10(PJ/PN)  (Sensors 2021 definition) :contentReference[oaicite:15]{index=15}
    PJ: mean(|j|^2) over the whole pulse
    """
    pj = float(np.mean(np.abs(j)**2))
    # pj should be > 0 as long as nf>=1 and nl ensures at least 1 cycle
    pn = pj / (10.0 ** (jnr_db / 10.0))

    noise = (rng.standard_normal(j.shape[0]) + 1j * rng.standard_normal(j.shape[0])) / np.sqrt(2.0)
    noise = noise.astype(np.complex64) * np.sqrt(pn).astype(np.float32)

    return j + noise


# =========================
# 4) 生成全量数据（严格 6000）
# =========================
def generate_dataset():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    s = make_baseband_lfm(FS, TP, KR)

    # Build parameter grid -> exactly 6000 samples:
    # For each NF: 25 (Nl) * 16 (JNR) * 5 (MC) = 2000
    samples = []
    for nf in NF_LIST:
        nl_list = NL_LIST_NF4 if int(nf) == 4 else NL_LIST_NF12
        for nl in nl_list:
            for jnr_db in JNR_DB_LIST:
                for mc in range(MC_REPEATS):
                    samples.append((int(nf), int(nl), int(jnr_db), int(mc)))

    assert len(samples) == 6000

    # Allocate arrays
    X = np.zeros((len(samples), 2, N), dtype=np.float32)     # IQ
    MASK = np.zeros((len(samples), N), dtype=np.uint8)
    TL_S = np.zeros((len(samples),), dtype=np.float32)
    TF_S = np.zeros((len(samples),), dtype=np.float32)
    NF_Y = np.zeros((len(samples),), dtype=np.int32)
    JNR_Y = np.zeros((len(samples),), dtype=np.int32)

    # Generate
    for idx, (nf, nl, jnr_db, mc) in enumerate(samples):
        j, mask = make_single_isrj(s, nl=nl, nf=nf)
        x = add_awgn_by_jnr(j, jnr_db=jnr_db, rng=rng)

        X[idx, 0, :] = x.real.astype(np.float32)
        X[idx, 1, :] = x.imag.astype(np.float32)
        MASK[idx, :] = mask

        tl = nl / FS
        tf = (nf * nl) / FS
        TL_S[idx] = np.float32(tl)
        TF_S[idx] = np.float32(tf)
        NF_Y[idx] = np.int32(nf)
        JNR_Y[idx] = np.int32(jnr_db)

    # Balanced split by NF: each NF has 2000 -> train 1400 / val 300 / test 300
    train_idx = []
    val_idx = []
    test_idx = []
    for nf in NF_LIST:
        idx_nf = np.where(NF_Y == nf)[0]
        rng.shuffle(idx_nf)

        n_total = idx_nf.shape[0]
        n_train = int(0.7 * n_total)   # 1400
        n_val = int(0.15 * n_total)    # 300
        n_test = n_total - n_train - n_val  # 300

        train_idx.append(idx_nf[:n_train])
        val_idx.append(idx_nf[n_train:n_train + n_val])
        test_idx.append(idx_nf[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)
    test_idx  = np.concatenate(test_idx)

    # Shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    meta = {
        "fs": FS,
        "tp": TP,
        "B": B,
        "Kr": KR,
        "N": N,
        "NF_LIST": NF_LIST.tolist(),
        "JNR_DB_LIST": JNR_DB_LIST.tolist(),
        "MC_REPEATS": MC_REPEATS,
        "Nl_list_nf12": NL_LIST_NF12.tolist(),
        "Nl_list_nf4": NL_LIST_NF4.tolist(),
        "seed": SEED,
        "label_definition": {
            "Tl": "slice width (seconds), Tl = Nl/fs",
            "NF": "forwarding times (integer), NF = M",
            "Tf": "forwarding width per cycle (seconds), Tf = NF*Tl",
            "Tu": "interrupted-sampling interval, Tu = (NF+1)*Tl"
        }
    }
    meta_json = json.dumps(meta, indent=2)

    def save_split(name: str, idxs: np.ndarray):
        np.savez_compressed(
            OUT_DIR / f"{name}.npz",
            X=X[idxs],
            mask=MASK[idxs],
            Tl_s=TL_S[idxs],
            Tf_s=TF_S[idxs],
            NF=NF_Y[idxs],
            JNR_dB=JNR_Y[idxs],
            meta_json=meta_json
        )

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

    print("Done.")
    print("Saved to:", OUT_DIR.resolve())
    print("Shapes:",
          "train", np.load(OUT_DIR/"train.npz")["X"].shape,
          "val",   np.load(OUT_DIR/"val.npz")["X"].shape,
          "test",  np.load(OUT_DIR/"test.npz")["X"].shape)


if __name__ == "__main__":
    generate_dataset()