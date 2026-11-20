"""Build the dashboard data bundle and the verified-claims tables.

Reads the raw CC0 Formula 1 CSVs in data/raw/, applies three documented grid
corrections (config/grid_corrections.csv), computes recorded and finish-cohort
position gains, and writes:

  docs/dashboard/assets/dashboard-data.js   data bundle loaded by the dashboard
  outputs/verified_claims.csv               headline numbers quoted in the video/report
  outputs/replay_coverage.csv               lap-coverage check for the six replays

Run from the repository root:  python src/build_dashboard_data.py
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config"
DEST = ROOT / "docs" / "dashboard" / "assets"
OUT = ROOT / "outputs"
DEST.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FIRST_YEAR, LAST_YEAR = 1950, 2024        # headline analysis stops at 2024
MODERN_FROM = 2010                        # default dashboard period
CIRCUIT_FROM, CIRCUIT_MIN_RACES = 1990, 10


def read(name):
    return pd.read_csv(RAW / f"{name}.csv", na_values=[r"\N"], low_memory=False)


results, races, drivers = read("results"), read("races"), read("drivers")
status, constructors, circuits = read("status"), read("constructors"), read("circuits")
laps = read("lap_times")

# ---------------------------------------------------------------- join
joined = results.merge(
    races[["raceId", "year", "round", "name", "date", "circuitId"]].rename(columns={"name": "race_name"}),
    on="raceId", validate="many_to_one")
joined = joined.merge(drivers[["driverId", "driverRef", "code", "forename", "surname"]],
                      on="driverId", validate="many_to_one")
joined["driver_name"] = joined.forename + " " + joined.surname
joined = joined.merge(status, on="statusId", validate="many_to_one")
joined = joined.merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor_name"}),
                      on="constructorId", validate="many_to_one")
joined = joined.merge(circuits[["circuitId", "circuitRef", "name", "country"]].rename(columns={"name": "circuit_name"}),
                      on="circuitId", validate="many_to_one")
joined = joined[joined.year.between(FIRST_YEAR, LAST_YEAR)].copy()
assert joined.resultId.is_unique
joined["raw_grid"] = joined.grid

# ------------------------------------------------ reviewed grid layer
corrections = pd.read_csv(CONFIG / "grid_corrections.csv")
reviewed = joined.copy()
for r in corrections.itertuples():
    mask = reviewed.raceId.eq(r.raceId) & reviewed.driverId.eq(r.driverId)
    assert mask.sum() == 1 and reviewed.loc[mask, "grid"].iloc[0] == r.raw_grid, r
    reviewed.loc[mask, "grid"] = r.reviewed_grid


def records(df, mapping):
    return json.loads(df[list(mapping)].rename(columns=mapping).to_json(orient="records", force_ascii=False))


def enrich(a):
    """Flag valid finishers and compute recorded vs finish-cohort gains."""
    a = a.copy()
    a["valid_grid"] = a.grid.gt(0)
    a["classified"] = a.position.notna()
    a["finish_status"] = (a.status.eq("Finished") | a.status.eq("Lapped")
                          | a.status.str.fullmatch(r"\+\d+ Laps?"))
    a["vf"] = a.valid_grid & a.classified & a.finish_status
    a["win"] = a.position.eq(1)
    a["podium"] = a.position.between(1, 3)
    v = a[a.vf].copy()
    v["og"] = v.grid - v.position                                # recorded gain
    v["ssr"] = v.groupby("raceId").grid.rank(method="min")       # start rank inside finish cohort
    v["sfr"] = v.groupby("raceId").position.rank(method="min")   # finish rank inside finish cohort
    v["ag"] = v.ssr - v.sfr                                      # finish-cohort gain
    v["ac"] = v.og - v.ag
    # Reviewed examples only, not an exhaustive pit-lane-start classifier.
    v["start_note"] = ""
    for rid, did in [(877, 20), (910, 1)]:
        v.loc[v.raceId.eq(rid) & v.driverId.eq(did), "start_note"] = \
            "Pit-lane start; positive grid value is only the archive encoding."
    return a, v


replay_spec = json.loads((CONFIG / "replays.json").read_text(encoding="utf-8"))


def replay_data(a, v):
    out = []
    for spec in replay_spec:
        focus = v[v.year.eq(spec["year"]) & v.driver_name.eq(spec["driver_name"]) & v.race_name.eq(spec["race_name"])]
        assert len(focus) == 1, spec["id"]
        f = focus.iloc[0]
        rid = int(f.raceId)
        field = a[a.raceId.eq(rid)].copy()
        field["label"] = field.code.fillna(field.surname.str[:3].str.upper())
        dup = field.label.duplicated(keep=False)
        field.loc[dup, "label"] = field.loc[dup, "label"] + " " + field.loc[dup, "driverId"].astype(str)
        assert field.label.is_unique
        rl = laps[laps.raceId.eq(rid)].merge(field[["driverId", "driver_name", "constructor_name", "label"]],
                                             on="driverId", validate="many_to_one")
        zero = field[field.grid.gt(0)].assign(lap=0, position=lambda x: x.grid)
        both = pd.concat([zero, rl], ignore_index=True).sort_values(["label", "lap"])
        fl = rl[rl.driverId.eq(f.driverId)].sort_values("lap")
        expected = int(f.laps)
        observed = fl.lap.astype(int).tolist()
        out.append({
            **{k: spec[k] for k in ["id", "year", "race_name", "driver_name", "key_laps"]},
            "driver_label": field.loc[field.driverId.eq(f.driverId), "label"].iloc[0],
            "grid": int(f.grid), "finish": int(f.position),
            "official_gain": int(f.og), "adjusted_gain": int(f.ag),
            "survivor_start_rank": int(f.ssr), "survivor_finish_rank": int(f.sfr),
            "expected_laps": expected, "observed_laps": len(fl),
            "coverage_complete": observed == list(range(1, expected + 1)),
            "last_lap_position": int(fl.position.iloc[-1]) if len(fl) else None,
            "key_laps": sorted({0, *spec["key_laps"], expected}),
            "laps": records(both, {"lap": "l", "position": "p", "label": "d",
                                   "driver_name": "dn", "constructor_name": "team"}),
        })
    # São Paulo 2024 is the default replay.
    return sorted(out, key=lambda x: x["id"] != "verstappen_2024_sao_paulo")


datasets, frames, scopes = {}, {}, []
for label, base in [("raw", joined), ("reviewed", reviewed)]:
    a, v = enrich(base)
    frames[label] = (a, v)
    datasets[label] = {
        "records": records(a[a.valid_grid], {"raceId": "r", "year": "y", "driverRef": "d", "driver_name": "dn",
                                             "circuitRef": "c", "circuit_name": "cn", "race_name": "rn",
                                             "grid": "g", "raw_grid": "rg", "position": "f",
                                             "positionOrder": "fo", "vf": "vf"}),
        "events": records(v[v.grid.ge(11) & v.position.le(3)],
                          {"raceId": "r", "year": "y", "date": "dt", "driverRef": "d", "driver_name": "dn",
                           "circuitRef": "c", "circuit_name": "cn", "race_name": "rn", "grid": "g",
                           "position": "f", "og": "og", "ag": "ag", "ac": "ac", "ssr": "ssr", "sfr": "sfr",
                           "start_note": "sn"}),
        "replays": replay_data(a, v),
    }
    for period, lo in [(f"{MODERN_FROM}–{LAST_YEAR}", MODERN_FROM), (f"{FIRST_YEAR}–{LAST_YEAR}", FIRST_YEAR)]:
        s = a[a.year.ge(lo)]
        pos = s[s.valid_grid]
        pole, p11, p17 = pos[pos.grid.eq(1)], pos[pos.grid.ge(11)], pos[pos.grid.eq(17)]
        scopes.append({
            "version": label, "period": period, "result_rows": len(s), "races": s.raceId.nunique(),
            "positive_grid_entries": len(pos), "unclassified_positive": int(pos.position.isna().sum()),
            "pole_wins": int(pole.win.sum()), "pole_entries": len(pole), "pole_conversion": pole.win.mean(),
            "p11_podiums": int(p11.podium.sum()), "p11_entries": len(p11), "p11_podium_rate": p11.podium.mean(),
            "p17_wins": int(p17.win.sum()), "p17_entries": len(p17),
        })

claims_df = pd.DataFrame(scopes)
claims_df.to_csv(OUT / "verified_claims.csv", index=False, encoding="utf-8-sig")
replay_coverage = pd.DataFrame([{k: r[k] for k in ["id", "driver_name", "year", "expected_laps", "observed_laps",
                                                   "coverage_complete", "last_lap_position", "finish"]}
                                for r in datasets["reviewed"]["replays"]])
replay_coverage.to_csv(OUT / "replay_coverage.csv", index=False, encoding="utf-8-sig")

# Circuit exploration: fixed 1990–2024, kept separate from the global filters.
a, v = frames["reviewed"]
cm = []
for cref, g in a[a.year.ge(CIRCUIT_FROM) & a.valid_grid].groupby("circuitRef"):
    n = g.raceId.nunique()
    if n < CIRCUIT_MIN_RACES:
        continue
    ev = g[g.vf & g.grid.ge(11) & g.position.le(3)].raceId.nunique()
    vg = v[v.circuitRef.eq(cref) & v.year.ge(CIRCUIT_FROM)]
    cm.append({"c": cref, "cn": g.circuit_name.iloc[0], "country": g.country.iloc[0], "n": int(n),
               "nf": float(1 - g.vf.mean()), "er": float(ev / n), "p90": float(vg.ag.quantile(.9))})
cdf = pd.DataFrame(cm)
payload = {
    "meta": {"title": "From Grid to Glory", "default_period": f"{MODERN_FROM}–{LAST_YEAR}",
             "historical_period": f"{FIRST_YEAR}–{LAST_YEAR}", "grid_corrections": len(corrections),
             "circuit_correlation": float(cdf.nf.corr(cdf.er)), "circuit_period": f"{CIRCUIT_FROM}–{LAST_YEAR}",
             "archive_max_year": int(races.year.max())},
    "datasets": datasets, "circuits": cm,
    "claims": json.loads(claims_df.to_json(orient="records")),
}
compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
(DEST / "dashboard-data.js").write_text("window.DASHBOARD_DATA=" + compact + ";\n", encoding="utf-8")

print(claims_df.to_string(index=False))
print(replay_coverage.to_string(index=False))
print(f"Wrote {DEST / 'dashboard-data.js'}")
