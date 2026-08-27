from metrics import (aggregate_trials, compute_trial_metrics, first_hold_time,
                     rank_algorithms, stop_and_go_transitions, yaw_flip_count)


def mk_tick(t, mode="SEGMENT_FOLLOW", ct=0.2, vert=0.1, pcmd=(0, 5, 2, 0),
            progress=None, **extra):
    return {"t": t, "mode": mode, "cross_track": ct, "vert_err": vert,
            "pcmd": list(pcmd), "route_progress": progress if progress is not None
            else min(1.0, t / 30.0), **extra}


def run_ticks(n=600, dt=0.05, **kw):
    return [mk_tick(i * dt, **kw) for i in range(n)]


def test_yaw_flip_count():
    assert yaw_flip_count([5, -5, 5, -5]) == 3
    assert yaw_flip_count([5, 5, 5]) == 0
    assert yaw_flip_count([5, -2, 5]) == 0          # small cmds ignored
    assert yaw_flip_count([2, -2, 2], min_mag=1) == 2


def test_stop_and_go_transitions():
    modes = ["SEGMENT_FOLLOW"] * 6
    assert stop_and_go_transitions([5, 0, 5, 0, 5, 0], modes) == 5
    assert stop_and_go_transitions([5, 5, 5, 5, 5, 5], modes) == 0
    # hover modes excluded
    assert stop_and_go_transitions([5, 0, 5], ["WAYPOINT_HOVER"] * 3) == 0


def test_first_hold_time():
    ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert first_hold_time(ts, [False, True, True, True, True, True], 3.0) == 1.0
    assert first_hold_time(ts, [True, True, False, True, True, False], 3.0) is None


def test_converging_trial_passes():
    ticks = []
    for i in range(1200):                            # 60 s at 20 Hz
        t = i * 0.05
        ct = max(0.15, 3.0 - 0.5 * t)                # converges by ~5.7 s
        mode = "SEGMENT_REJOIN" if ct > 0.5 else "SEGMENT_FOLLOW"
        ticks.append(mk_tick(t, mode=mode, ct=ct, pcmd=(0, 6, 1, 0),
                             progress=min(1.0, t / 40.0), transitions=1))
    ticks[-1]["mode"] = "COMPLETED"
    m = compute_trial_metrics(ticks)
    assert m["pass"] and m["converged"] and m["follow_reached"]
    assert m["completed_route"]
    assert 4.0 < m["time_to_converge_s"] < 6.0
    assert m["corridor_time_ratio"] > 0.95
    assert "completed_route" in m["labels"]


def test_never_converging_trial_fails():
    m = compute_trial_metrics(run_ticks(ct=2.5))
    assert not m["pass"] and not m["converged"]
    assert "never_converged" in m["failure_reasons"]


def test_oscillation_failure_detected():
    ticks = run_ticks(ct=0.3)
    for i, tk in enumerate(ticks):
        tk["pcmd"] = [0, 5, 15 if i % 2 == 0 else -15, 0]
    m = compute_trial_metrics(ticks)
    assert m["yaw_flips_per_min"] > 20
    assert "oscillation_failure" in m["labels"]
    assert not m["pass"]


def test_stop_and_go_failure_detected():
    ticks = run_ticks(ct=1.2, mode="SEGMENT_REJOIN", progress=0.1)
    for tk in ticks:
        tk["pcmd"] = [0, 0, 2, 0]                    # pitch stays zero
    m = compute_trial_metrics(ticks)
    assert m["pitch_zero_ratio"] > 0.9
    assert "stop_and_go_failure" in m["labels"]


def test_abort_labels():
    ticks = run_ticks(n=100, ct=0.3)
    ticks.append(mk_tick(5.0, mode="ABORT_OR_MANUAL", ct=6.5,
                         abort_reason="horizontal_hard_abort", pcmd=(0, 0, 0, 0)))
    m = compute_trial_metrics(ticks)
    assert m["aborted"] and "diverged" in m["labels"]
    ticks[-1]["abort_reason"] = "vertical_profile_failure"
    m = compute_trial_metrics(ticks)
    assert "vertical_profile_failure" in m["labels"]
    ticks[-1]["abort_reason"] = "telemetry_lost"
    m = compute_trial_metrics(ticks)
    assert "telemetry_safety_block" in m["labels"]


def test_trivial_start_flagged():
    m = compute_trial_metrics(run_ticks(ct=0.2))
    assert m["trivial_start_on_path"]
    assert "trivial_start_on_path" in m["labels"]
    assert "trivial_start_on_path" in m["failure_reasons"]
    assert not m["pass"]


def test_early_late_segment_switch_counted():
    ticks = run_ticks(n=200, ct=0.2)
    ticks[50]["segment_switch"] = "arrived"
    ticks[50]["switch_d_end"] = 1.5                  # switched far from the end
    ticks[50]["switch_progress"] = 0.5
    ticks[150]["segment_switch"] = "forced_timeout"
    m = compute_trial_metrics(ticks)
    assert m["early_segment_switches"] == 1
    assert m["late_segment_switches"] == 1
    assert "early_segment_switch" in m["labels"]
    assert "late_segment_switch" in m["labels"]


def test_aggregation_and_ranking():
    def trial(algo, ok, yaw, pert="clean"):
        ticks = run_ticks(n=400, ct=0.2 if ok else 2.5,
                          pcmd=(0, 6, 1, 0), transitions=1)
        if ok:
            for i, tick in enumerate(ticks[:40]):
                tick["cross_track"] = max(0.2, 1.0 - i * 0.02)
            ticks[-1]["mode"] = "COMPLETED"
        return {"algorithm": algo, "yaw_error_deg": yaw, "perturbation": pert,
                "metrics": compute_trial_metrics(ticks)}

    trials = ([trial("good", True, 10)] * 8 + [trial("good", False, 90)] * 2 +
              [trial("bad", False, 10)] * 6 + [trial("bad", True, 10, "noise0.2m")] * 4)
    s = aggregate_trials(trials)
    assert s["by_algorithm"]["good"]["pass_rate"] == 0.8
    assert s["by_algorithm"]["bad"]["pass_rate"] == 0.4
    assert s["by_algorithm"]["good"]["pass_rate_by_abs_yaw_deg"]["10"] == 1.0
    assert s["by_algorithm"]["good"]["pass_rate_by_abs_yaw_deg"]["90"] == 0.0
    assert "noise0.2m" in s["by_algorithm"]["bad"]["pass_rate_by_perturbation"]
    ranked = rank_algorithms(s)
    assert ranked[0][0] == "good"


def test_zero_cross_track_error_ranks_better_than_nonzero_error():
    def row(cross_track):
        return {
            "n": 1,
            "pass_rate": 1.0,
            "route_completion_rate": 1.0,
            "corridor_time_ratio_mean": 1.0,
            "cross_track_mean_m": cross_track,
            "yaw_flips_per_min_mean": 0.0,
            "stop_and_go_score_mean": 0.0,
            "hard_aborts": 0,
        }

    ranked = rank_algorithms({
        "by_algorithm": {"perfect": row(0.0), "imperfect": row(0.1)}
    })
    assert ranked[0][0] == "perfect"
