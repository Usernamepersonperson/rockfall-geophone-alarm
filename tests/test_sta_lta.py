import dataclasses

import numpy as np

from rockfall.config import DetectorCfg
from rockfall.detect.sta_lta import detect_triggers, recursive_sta_lta

FS = 100
CFG = DetectorCfg(sta_s=0.5, lta_s=10.0, on=4.0, off=2.0, freeze_lta=True, max_trigger_s=600,
                  merge_gap_s=1.0, min_trigger_s=0.2)


def noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


def test_ratio_matches_naive_recursion() -> None:
    x = noise(2000)
    ns, nl = 50, 400
    sta = lta = 0.0
    ref = []
    for v in x:
        sta += (v * v - sta) / ns
        lta += (v * v - lta) / nl
        ref.append(sta / (lta + 1e-12))
    got = recursive_sta_lta(x, ns, nl)
    np.testing.assert_allclose(got[nl:], np.array(ref)[nl:], rtol=1e-9)
    assert np.all(got[:nl] == 0)


def test_step_triggers_shortly_after_onset_and_ends() -> None:
    x = noise(6000)
    x[3000:3500] *= 10.0  # 5 s of 10x amplitude starting at t = 30 s
    trg = detect_triggers(x, FS, CFG)
    assert len(trg) == 1
    assert 30.0 <= trg[0].on_s <= 30.6
    assert 35.0 <= trg[0].off_s <= 37.5
    assert trg[0].peak_ratio > 20 and not trg[0].ongoing


def test_single_impulse_triggers() -> None:
    x = noise(6000)
    x[4000] = 80.0
    trg = detect_triggers(x, FS, CFG)
    assert len(trg) == 1 and abs(trg[0].on_s - 40.0) < 0.1


def test_pure_noise_does_not_trigger() -> None:
    assert detect_triggers(noise(30000, seed=3), FS, CFG) == []


def test_warmup_is_ignored() -> None:
    x = noise(1500)
    x[:200] *= 20.0  # loud start while the LTA is still empty
    assert detect_triggers(x, FS, CFG) == []


def test_freeze_keeps_long_emergent_event_triggered() -> None:
    x = noise(30000)
    x[8000:20000] *= 6.0  # 120 s rumble, much longer than the 10 s LTA
    frozen = detect_triggers(x, FS, CFG)
    free = detect_triggers(x, FS, dataclasses.replace(CFG, freeze_lta=False))
    assert len(frozen) == 1 and frozen[0].off_s - frozen[0].on_s > 115
    assert free[0].off_s - free[0].on_s < 30  # without freezing the LTA catches up and the trigger drops out


def test_close_triggers_are_merged() -> None:
    x = noise(8000)
    x[3000:3100] *= 12.0
    x[3600:3700] *= 12.0  # second burst ~5 s after the first one ended
    assert len(detect_triggers(x, FS, dataclasses.replace(CFG, merge_gap_s=8.0))) == 1
    assert len(detect_triggers(x, FS, dataclasses.replace(CFG, merge_gap_s=1.0))) == 2


def test_event_still_running_at_end_of_record_is_flagged() -> None:
    x = noise(6000)
    x[5000:] *= 10.0
    (trg,) = detect_triggers(x, FS, CFG)
    assert trg.ongoing
