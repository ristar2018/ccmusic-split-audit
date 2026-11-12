"""Full audit of the compute ledger (review comment C7): find every run that cost.json does not count.

Read-only over `logs/*.log`, `runs*/`, `data/results/cost.json`; writes `data/results/human_review/compute_ledger.{json,csv}`.
No extrapolation: minutes are only the integer minutes written into the driver logs; a missing log is flagged as missing.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "results", "human_review")
LOGS = os.path.join(ROOT, "logs")

RE_DRV = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+(A\d)\s+(\S+)\s+rc=(-?\d+)\s+(\d+)min")
RE_SKIP = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+(A\d)\s+(\S+)\s+SKIP")
RE_C1 = re.compile(r"^\[(C1|B1)\]\s+(?:(\S+)\s+)?(A1r|C1)\s+(\S+)\s+(seed\d+\s+)?rc=(-?\d+)\s+(\d+)min")


def parse(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s:
            continue
        m = RE_DRV.match(s)
        if m:
            rows.append({"file": os.path.basename(path), "idx": int(m.group(1)), "total": int(m.group(2)),
                         "lib": m.group(3), "arm": m.group(4), "bb": m.group(5),
                         "rc": int(m.group(6)), "minutes": int(m.group(7)), "skipped": False})
            continue
        m = RE_SKIP.match(s)
        if m:
            rows.append({"file": os.path.basename(path), "idx": int(m.group(1)), "total": int(m.group(2)),
                         "lib": m.group(3), "arm": m.group(4), "bb": m.group(5),
                         "rc": 0, "minutes": 0, "skipped": True})
            continue
        m = RE_C1.match(s)
        if m:
            rows.append({"file": os.path.basename(path), "idx": None, "total": None,
                         "lib": m.group(2) or "CTIS", "arm": m.group(3), "bb": m.group(4),
                         "rc": int(m.group(6)), "minutes": int(m.group(7)), "skipped": False,
                         "seed": (m.group(5) or "").strip()})
    return rows


def sealed_groups(root):
    d = os.path.join(ROOT, root, "sealed_eval")
    if not os.path.isdir(d):
        return []
    out = []
    for g in sorted(os.listdir(d)):
        p = os.path.join(d, g)
        if not os.path.isdir(p):
            continue
        seeds = len(glob.glob(os.path.join(p, "seed*_qtest.npz")))
        out.append({"group": g, "seeds": seeds})
    return out


def p3_folds():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs_p3", "*", "*", "seed*", "fold*_result.json"))):
        j = json.load(open(p, encoding="utf-8"))
        rows.append({"arm": j.get("arm"), "bb": j.get("bb"), "seed": j.get("seed"),
                     "fold": j.get("fold"), "elapsed_s": float(j.get("elapsed_s") or 0)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    fams = {
        "E0_smoke": ["e0_ctis_a0_alexnet.log"],
        "E1b_A0A1": ["e1b_driver.log"],
        "E1c_A2": ["e1c_driver.log", "e1c_wrapper.log"],
        "E1f_negcontrol": ["e1f_driver.log", "e1f_driver_1.log", "e1f_drv_driver.log", "e1f2_driver.log"],
        "B2_negcontrol_rerun": ["b2_driver.log", "b2_nohup.log"],
        "B1C1_probe_and_C1": ["b1c1_driver.log"],
    }
    ledger = {"families": {}, "sealed": {}, "p3": {}, "flags": []}

    for fam, files in fams.items():
        rows = []
        for f in files:
            rows += parse(os.path.join(LOGS, f))
        # de-duplication: when the same (lib,arm,bb) appears in several logs, keep the last non-skipped row
        best = {}
        for r in rows:
            key = (r["lib"], r["arm"], r["bb"], r.get("seed", ""))
            if key not in best or (best[key]["skipped"] and not r["skipped"]):
                best[key] = r
        rows = list(best.values())
        minutes = sum(r["minutes"] for r in rows if not r["skipped"])
        ledger["families"][fam] = {
            "log_lines": len(rows),
            "runs": len(rows),
            "ok": sum(1 for r in rows if r["rc"] == 0 and not r["skipped"]),
            "failed": sum(1 for r in rows if r["rc"] != 0),
            "skipped_already_done": sum(1 for r in rows if r["skipped"]),
            "zero_minute_runs": sum(1 for r in rows if not r["skipped"] and r["minutes"] == 0),
            "minutes_sum": minutes,
            "gpu_h_from_logs": round(minutes / 60.0, 3),
            "files": files,
            "detail": [{"file": r["file"], "arm": r["arm"], "bb": r["bb"],
                        "seed": r.get("seed", ""), "minutes": r["minutes"],
                        "skipped": r["skipped"]} for r in rows],
        }
        if ledger["families"][fam]["zero_minute_runs"]:
            ledger["flags"].append(
                "%s: %d runs logged 0 minutes in the driver log (rounding); machine time is underestimated"
                % (fam, ledger["families"][fam]["zero_minute_runs"]))

    for root in ("runs", "runs_chk", "runs_b2", "runs_p3"):
        ledger["sealed"][root] = sealed_groups(root)

    p3 = p3_folds()
    by_arm = {}
    for r in p3:
        by_arm.setdefault(r["arm"], {"folds": 0, "groups": set(), "gpu_h": 0.0})
        by_arm[r["arm"]]["folds"] += 1
        by_arm[r["arm"]]["groups"].add((r["bb"], r["seed"]))
        by_arm[r["arm"]]["gpu_h"] += r["elapsed_s"] / 3600.0
    ledger["p3"] = {
        "total_folds": len(p3),
        "total_gpu_h": round(sum(r["elapsed_s"] for r in p3) / 3600.0, 3),
        "by_arm": {k: {"folds": v["folds"], "gpu_h": round(v["gpu_h"], 3),
                       "n_groups": len(v["groups"])} for k, v in sorted(by_arm.items())},
    }

    cost = json.load(open(os.path.join(ROOT, "data", "results", "cost.json"), encoding="utf-8"))
    covered = {
        "E1b": cost["E1b"]["gpu_h"], "E1c": cost["E1c"]["gpu_h"],
        "E1f_negcontrol": cost["E1f"]["gpu_h"], "P3": cost["P3"]["gpu_h"],
    }
    measured = {
        "E1b": ledger["families"]["E1b_A0A1"]["gpu_h_from_logs"],
        "E1c": ledger["families"]["E1c_A2"]["gpu_h_from_logs"],
        "E1f_negcontrol": ledger["families"]["E1f_negcontrol"]["gpu_h_from_logs"]
                          + ledger["families"]["B2_negcontrol_rerun"]["gpu_h_from_logs"],
        "P3": ledger["p3"]["total_gpu_h"],
    }
    uncovered = {
        "B1C1_probe_and_C1": ledger["families"]["B1C1_probe_and_C1"]["gpu_h_from_logs"],
        "E0_smoke": ledger["families"]["E0_smoke"]["gpu_h_from_logs"],
    }
    ledger["cost_json"] = cost
    # only the sum of the components is the corrected value: the P3 component in cost_json_total is already below the measured value, so the old total cannot be used as a base for the increment
    b1rows = ledger["families"]["B1C1_probe_and_C1"]["detail"]
    c1_min = sum(r["minutes"] for r in b1rows if r["arm"] == "C1" and not r["skipped"])
    a1r_min = sum(r["minutes"] for r in b1rows if r["arm"] == "A1r" and not r["skipped"])
    components = {
        "E1b_confirmatory": measured["E1b"],
        "E1c_A2_random_split": measured["E1c"],
        "B2_negative_controls": measured["E1f_negcontrol"],
        "P3_cross_instrument": measured["P3"],
        "C1_attribution": c1_min / 60.0,
        "A1r_probes": a1r_min / 60.0,
    }
    ledger["reconciliation"] = {
        "cost_json_total": cost["total_measured_gpu_h"],
        "covered_by_family": covered,
        "measured_by_family": measured,
        "components_gpu_h": {k: round(v, 4) for k, v in components.items()},
        "component_sum_gpu_h": round(sum(components.values()), 4),
        "not_in_cost_json": uncovered,
        "not_in_cost_json_gpu_h": round(sum(uncovered.values()), 3),
        "corrected_total_gpu_h": round(sum(components.values()), 4),
        "classification": {
            "confirmatory": {"E1b_confirmatory": round(components["E1b_confirmatory"], 4)},
            "additional": {k: round(components[k], 4) for k in
                           ("E1c_A2_random_split", "B2_negative_controls", "P3_cross_instrument")},
            "supplementary": {k: round(components[k], 4) for k in ("C1_attribution", "A1r_probes")},
            "abandoned_unquantified": [
                "logs/e1f_chest_falsetto_A2_alexnet.log is 0 bytes (one failed attempt, no timing recorded)",
                "no log at all under logs/ for the lambda=0.1 trial; the failed attempt cannot be quantified",
            ],
            "excluded_but_known": ["E0 smoke (e0_ctis_a0_alexnet.log)", "wall clock of the B2 negative-control rerun (driver log rounds to 0)"],
        },
        "runs_not_counted": {
            "C1": ledger["families"]["B1C1_probe_and_C1"]["runs"],
            "E1f_skipped_lines": ledger["families"]["E1f_negcontrol"]["skipped_already_done"],
            "p3_folds_vs_expected": [ledger["p3"]["total_folds"], cost["P3"]["expected_folds"]],
        },
    }
    ledger["flags"].append("cost.json covers only the four families E1b/E1c/E1f/P3; C1 (runs_chk) and the A1r probes (b1c1) are not accounted")
    ledger["flags"].append("of the 10 lines logged for E1f, %d are SKIP (already done); 12 groups in total"
                           % ledger["families"]["E1f_negcontrol"]["skipped_already_done"])
    ledger["flags"].append("P3 accounts for %d folds while expected_folds says 108 (36 groups x 3 folds); the calibers disagree"
                           % ledger["p3"]["total_folds"])

    json.dump(ledger, open(os.path.join(OUT, "compute_ledger.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "compute_ledger.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("family,runs,ok,failed,skipped,zero_minute_runs,minutes,gpu_h,in_cost_json\n")
        for fam, v in ledger["families"].items():
            inc = {"E1b_A0A1": "E1b", "E1c_A2": "E1c", "E1f_negcontrol": "E1f"}.get(fam, "NO")
            f.write("%s,%d,%d,%d,%d,%d,%d,%.3f,%s\n" % (
                fam, v["runs"], v["ok"], v["failed"], v["skipped_already_done"],
                v["zero_minute_runs"], v["minutes_sum"], v["gpu_h_from_logs"], inc))
        for arm, v in ledger["p3"]["by_arm"].items():
            f.write("P3_%s,%d,0,0,0,0,0,%.3f,P3\n" % (arm, v["folds"], v["gpu_h"]))

    print(json.dumps(ledger["reconciliation"], ensure_ascii=False, indent=1))
    for fl in ledger["flags"]:
        print("FLAG:", fl)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
