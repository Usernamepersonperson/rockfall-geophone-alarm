"""Command line: ``python -m rockfall simulate|bench``."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from rockfall.bench import format_report, run_bench
from rockfall.config import load_config
from rockfall.pipeline import Pipeline, RunResult
from rockfall.sim.generator import Scenario, simulate
from rockfall.sim.scenarios import NAMES, named_scenario
from rockfall.sinks import JsonlSink, MqttSink, Sink, StdoutSink, WebhookSink
from rockfall.site import demo_site


def _print_summary(sc: Scenario, run: RunResult, overhead_s: float) -> None:
    print(f"\nscenario '{sc.name}' seed={sc.seed} duration={sc.duration_s:.0f}s")
    for ev in sc.events:
        print(f"  truth: {ev.kind:<10} {ev.t_start:7.1f}s - {ev.t_end:7.1f}s")
    for f in sc.faults:
        print(f"  fault: {f.station} {f.kind} from {f.t_start:.0f}s")
    print("\nstation health (last window):")
    print("  " + "  ".join(f"{k}:{v[-1][1].status}" for k, v in run.health.items()))
    print(f"\nassociated events ({len(run.events)}):")
    for e in run.events:
        print(f"  #{e.event_id} t={e.first_on_s:7.1f}s stations={','.join(e.stations)} speed={e.speed_ms:.0f}m/s "
              f"cos_downslope={e.cos_downslope:+.2f} -> {e.label}" + (f" [{e.rule}]" if e.rule else ""))
    print(f"\nfixed latency overhead per decision: {overhead_s:.1f}s")
    print("final state:", run.final_state.value)


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.require_ack:
        from dataclasses import replace
        cfg = replace(cfg, alarm=replace(cfg.alarm, require_ack=True))
    site = demo_site()
    sinks: list[Sink] = [StdoutSink()]
    if args.audit_log:
        sinks.append(JsonlSink(args.audit_log))
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        sinks.append(MqttSink(host, int(port or 1883), site=cfg.site))
    if args.webhook:
        sinks.append(WebhookSink(args.webhook))
    sc = named_scenario(args.scenario, args.seed)
    print("state transitions (time = decision time incl. latency overhead):")
    run = Pipeline(cfg, site, sinks).run(simulate(sc, site), ack_times=args.ack_at or [])
    _print_summary(sc, run, cfg.latency.fixed_overhead_s)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(format_report(run_bench(cfg, demo_site(), n=args.n, seed=args.seed, progress=True)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rockfall", description="Geophone alarm chain demo (synthetic data only).")
    ap.add_argument("--config", help="path to a thresholds JSON (default: built-in)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="run one scripted scenario through the whole chain")
    s.add_argument("--scenario", choices=NAMES, default="rockfall")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--audit-log", help="append hash-chained JSONL audit records here")
    s.add_argument("--mqtt", metavar="HOST[:PORT]", help="publish to hazard/<site>/alarm (needs paho-mqtt)")
    s.add_argument("--webhook", metavar="URL", help="POST every transition as JSON")
    s.add_argument("--require-ack", action="store_true", help="RED needs a manual acknowledgement to clear")
    s.add_argument("--ack-at", type=float, action="append", metavar="SECONDS", help="simulate an operator ack at this time")
    s.set_defaults(func=cmd_simulate)
    b = sub.add_parser("bench", help="many randomised scenarios: confusion matrix, false alarms, latency")
    b.add_argument("--n", type=int, default=120)
    b.add_argument("--seed", type=int, default=1)
    b.set_defaults(func=cmd_bench)
    args = ap.parse_args(argv)
    return int(args.func(args))
