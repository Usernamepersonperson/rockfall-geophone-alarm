import numpy as np
import pytest

from rockfall.config import AssociationCfg
from rockfall.detect.associate import associate, fit_arrivals
from rockfall.detect.sta_lta import Trigger
from rockfall.site import Site, Station, demo_site

CFG = AssociationCfg(min_stations=3, max_span_s=150, rms_tol_base_s=1.0, rms_tol_frac=0.1,
                     min_speed_ms=1.0, max_speed_ms=150.0, min_cos_downslope=0.5)
DOWN = (0.0, -1.0)


def trig(station: str, t: float) -> Trigger:
    return Trigger(station, t, t + 5, False, 10.0)


def test_downslope_front_speed_recovered_1d() -> None:
    xy = demo_site().xy()
    t = 100.0 + (250.0 - xy[:, 1]) / 8.0  # front moving downhill at 8 m/s
    fit = fit_arrivals(xy, t, DOWN)
    assert fit.speed_ms == pytest.approx(8.0, rel=0.02)
    assert fit.cos_downslope == pytest.approx(1.0, abs=1e-6)
    assert fit.rms_s < 1e-6 and fit.plausible(CFG)


def test_upslope_front_is_not_plausible() -> None:
    xy = demo_site().xy()
    fit = fit_arrivals(xy, 100.0 + xy[:, 1] / 8.0, DOWN)
    assert fit.cos_downslope == pytest.approx(-1.0, abs=1e-6)
    assert not fit.plausible(CFG)


def test_simultaneous_arrivals_are_too_fast_to_be_plausible() -> None:
    xy = demo_site().xy()
    fit = fit_arrivals(xy, np.full(6, 50.0), DOWN)
    assert fit.speed_ms > 1e5 and not fit.plausible(CFG)


def test_two_dimensional_array_recovers_azimuth() -> None:
    xy = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50.0]])
    direction = np.array([1.0, -1.0]) / np.sqrt(2)  # toward south-east
    t = 5.0 + (xy @ direction) / 20.0
    fit = fit_arrivals(xy, t, DOWN)
    assert fit.speed_ms == pytest.approx(20.0, rel=1e-6)
    assert fit.cos_downslope == pytest.approx(np.sqrt(0.5), abs=1e-6)
    assert fit.dof == 2  # 3-parameter fit used


def test_timing_noise_gives_nonzero_residual() -> None:
    xy = demo_site().xy()
    t = 100.0 + (250.0 - xy[:, 1]) / 10.0 + np.array([0.0, 0.6, -0.5, 0.4, -0.3, 0.0])
    fit = fit_arrivals(xy, t, DOWN)
    assert 0.1 < fit.rms_s < 1.0 and fit.speed_ms == pytest.approx(10.0, rel=0.1)


def test_needs_k_stations() -> None:
    assert associate([trig("G1", 10.0), trig("G2", 15.0)], demo_site(), CFG) == []


def test_k_consistent_stations_form_an_event_at_the_kth_trigger() -> None:
    trg = [trig(f"G{i + 1}", 100 + 5.0 * i) for i in range(4)]  # 50 m / 5 s = 10 m/s
    (ev,) = associate(trg, demo_site(), CFG)
    assert ev.t_assoc == 110.0  # third station completes the event
    assert sorted(ev.members) == ["G1", "G2", "G3", "G4"]  # G4 is absorbed
    assert ev.fit.speed_ms == pytest.approx(10.0, rel=1e-3)
    assert ev.fit0 is not None and ev.fit0.n == 3


def test_inconsistent_outlier_is_rejected() -> None:
    trg = [trig("G1", 100), trig("G2", 105), trig("G3", 110), trig("G5", 40.0)]  # G5 fires at a random time
    (ev,) = associate(trg, demo_site(), CFG)
    assert "G5" not in ev.members


def test_triggers_outside_time_window_do_not_associate() -> None:
    assert associate([trig("G1", 0), trig("G2", 200), trig("G3", 400)], demo_site(), CFG) == []


def test_one_noisy_station_alone_never_associates() -> None:
    site = Site("s", (Station("A", 0, 0), Station("B", 0, -50), Station("C", 0, -100)))
    assert associate([trig("A", 1), trig("A", 30), trig("A", 60)], site, CFG) == []
