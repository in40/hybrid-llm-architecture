"""
Frozen pre-registration statistics (PREREG-arms-and-stats.md §4). Pure stdlib.
Commit BEFORE any hybrid results are observed; any change afterwards = revision.

Primary: paired per-seed relative gain g_s = (BPB_A0(s) - BPB_A1(s)) / BPB_A0(s).
CI: mean +/- t(0.975, df=n-1) * s_d / sqrt(n). df=2 at n=3; df=4 if n=5 expansion.
Gate (ALL required):
  1) CI_lower(mean g) >= 1.0%
  2) |FLOPs_A1 - FLOPs_A0| / FLOPs_A0 <= 0.5%
  3) throughput_A1 >= 0.90 * throughput_A0 (bytes/s, solo window)
  4) peak VRAM_A1 <= 1.10 * VRAM_A0
  5) V-B veto NOT triggered: triggers iff rel drop > 2% AND abs drop > 0.01,
     abs condition only if baseline mean F1 > 0.05 (else veto uninformative).
Expansion rule: s_d over the 3 pairs > 0.002 bpb -> add seeds 789,1024 (n=5).
"""
import math

T_CRIT = {2: 4.302938, 4: 2.776445}  # t(0.975, df)


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_gain(bpb_a0, bpb_a1):
    """bpb lists aligned by seed id."""
    assert len(bpb_a0) == len(bpb_a1) and len(bpb_a0) in (3, 5)
    return [(a - b) / a for a, b in zip(bpb_a0, bpb_a1)]


def primary_stats(gains):
    n = len(gains)
    m = mean(gains)
    sd = stdev(gains)
    half = T_CRIT[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "s_d": sd, "ci_lo": m - half, "ci_hi": m + half}


def expansion_needed(bpb_a0, bpb_a1):
    return stdev([a - b for a, b in zip(bpb_a0, bpb_a1)]) > 0.002


def vb_veto_triggers(base_f1, hyb_f1):
    rel = (base_f1 - hyb_f1) / base_f1 if base_f1 > 0 else 0.0
    absd = base_f1 - hyb_f1
    if base_f1 <= 0.05:
        return None  # uninformative, documented
    return rel > 0.02 and absd > 0.01


def gate(bpb_a0, bpb_a1, flops_a0, flops_a1, tp_a0, tp_a1, vram_a0, vram_a1,
         vb_base_f1, vb_hyb_f1):
    st = primary_stats(paired_gain(bpb_a0, bpb_a1))
    veto = vb_veto_triggers(vb_base_f1, vb_hyb_f1)
    checks = {
        "ci_lower_ge_1pct": st["ci_lo"] >= 0.01,
        "flops_within_0p5pct": abs(flops_a1 - flops_a0) / flops_a0 <= 0.005,
        "throughput_ge_0p9x": tp_a1 >= 0.90 * tp_a0,
        "vram_le_1p1x": vram_a1 <= 1.10 * vram_a0,
        "vb_veto_ok": veto in (False, None),
    }
    return {"stats": st, "checks": checks, "vb_veto": veto,
            "h1_supported": all(checks.values()),
            "wording": ("H1 supported" if all(checks.values())
                        else "not supported: insufficient evidence of >=1% gain"
                        if checks["ci_lower_ge_1pct"] is False else "not supported")}


if __name__ == "__main__":
    # self-tests (synthetic)
    g = gate([0.9182, 0.9184, 0.9198], [0.9000, 0.9010, 0.9020],
             2.28e18, 2.28e18 * 1.0002, 100.0, 95.0, 40.0, 42.0, 0.42, 0.415)
    assert g["stats"]["n"] == 3 and g["h1_supported"], g
    # gain ~2% with tiny spread -> pass
    g2 = gate([0.9182, 0.9184, 0.9198], [0.9180, 0.9182, 0.9181],
              2.28e18, 2.28e18, 100.0, 80.0, 40.0, 40.0, 0.42, 0.42)
    assert not g2["checks"]["throughput_ge_0p9x"] and not g2["h1_supported"]
    # veto: rel 2.125% but abs 0.0085 -> NOT triggered (AND-logic)
    assert vb_veto_triggers(0.40, 0.3915) is False
    # veto: rel 3% AND abs 0.012 -> triggered
    assert vb_veto_triggers(0.40, 0.388) is True
    # veto: baseline near floor -> uninformative
    assert vb_veto_triggers(0.04, 0.03) is None
    # expansion rule: s_d 0.0025 > 0.002 -> True
    assert expansion_needed([0.918, 0.9184, 0.9198], [0.9155, 0.9184, 0.9220])
    print("prereg_stats self-tests PASS")
