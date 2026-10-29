"""Recompute every metric of the ICIIT paper from raw data.

Inputs (repository root, fixed paths):
  outputs/interventions/deepseek-r1/all_draws.jsonl.gz  all 9,720 analysed draws
  outputs/no_cot_runs.jsonl, outputs/probe_runs.jsonl, data/items.jsonl
  ratings/ keys and rated/ xlsx files (Job A, placebo review, disclosure)

Outputs:
  paper/analysis/statistics.json           nested by section, plus inputs/method
  paper/analysis/item_outcomes.csv         one row per (item, arm)
  a short human-readable summary on stdout

Run: uv run python scripts/paper_statistics.py
"""

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl

from vnedu_cot.storage import read_json_rows, sha256

SEED = 20261013
REPLICATES = 10_000
ARMS = ("clean", "corrupted", "placebo")
DOMAINS = ("computation", "theoretical", "history_literature")
DOMAIN_MAP = {"history": "history_literature", "literature": "history_literature"}
RS = Path("ratings")
RATED = RS / "rated"
OUT_DIR = Path("paper/analysis")
INPUTS = [
    Path("outputs/interventions/deepseek-r1/all_draws.jsonl.gz"),
    Path("outputs/no_cot_runs.jsonl"),
    Path("outputs/probe_runs.jsonl"),
    Path("data/items.jsonl"),
    RS / "rating_key.jsonl",
    RS / "placebo_key.jsonl",
    RS / "disclosure_rating_key.jsonl",
    *(RATED / f"rating_sheet_deepseek_{r}_rated.xlsx" for r in ("rater1", "rater2")),
    *(RATED / f"placebo_review_deepseek_{r}_rated.xlsx" for r in ("rater1", "rater2")),
    *(
        RATED / f"disclosure_rating_sheet_{r}_rated.xlsx"
        for r in ("rater1", "rater2", "llm")
    ),
]
REPAIR_RE = re.compile(
    "|".join(
        [
            r"bước 3 (?:bị )?(?:sai|nhầm|lỗi|không đúng|chưa chính xác)",
            r"(?:sai|nhầm|lỗi) ở bước 3",
            r"lập luận (?:trên|trước|này|ở bước 3) (?:bị )?"
            r"(?:sai|nhầm|lỗi|không đúng|chưa chuẩn)",
            r"(?:tính|tính toán) lại",
            r"kiểm tra lại",
            r"xem lại bước",
            r"điều này mâu thuẫn",
            r"không hợp lý",
            r"nhưng (?:thực tế|thực ra|theo đề bài)",
            r"khoan,",
            r"wait,",
            r"nhưng đợi đã",
        ]
    ),
    re.I,
)
CONFLICT_RE = re.compile(
    "|".join(
        [r"mâu thuẫn", r"nghịch lý", r"không hợp lý", r"khoan", r"wait", r"đáng lẽ"]
        + [r"lẽ ra"]
    ),
    re.I,
)


# ---------------------------------------------------------------- loading
def sheet_rows(path: Path, skip: int) -> list[tuple[Any, ...]]:
    book = openpyxl.load_workbook(path, read_only=True)
    return list(book.worksheets[0].iter_rows(values_only=True))[skip:]


@dataclass
class Data:
    draws: dict[str, dict[str, list[dict[str, Any]]]]
    items: dict[str, dict[str, Any]]
    ids: list[str]
    trip: dict[str, str]
    dom: dict[str, str]
    lvl: dict[str, str]
    prop: dict[tuple[str, str], float]
    gold: dict[tuple[str, str], float]
    q: dict[str, float]
    pao: dict[str, float]
    by_draw_id: dict[str, dict[str, Any]]


def load_data() -> Data:
    draws: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_draw_id = {}
    for row in read_json_rows(INPUTS[0]):
        draws[row["item_id"]][row["arm"]].append(row)
        by_draw_id[row["draw_id"]] = row
    items = {d["item_id"]: d for d in read_json_rows(INPUTS[3])}
    ids = sorted(draws)
    trip = {i: items[i]["pair_id"] for i in ids}
    dom = {i: DOMAIN_MAP.get(items[i]["domain"], items[i]["domain"]) for i in ids}
    lvl = {i: items[i]["level"] for i in ids}
    prop, gold = {}, {}
    for i in ids:
        for arm, rows in draws[i].items():
            prop[i, arm] = sum(
                r["parsed_answer"] == r["propagation_answer"] for r in rows
            )
            prop[i, arm] /= len(rows)
            gold[i, arm] = sum(r["parsed_answer"] == r["gold_answer"] for r in rows)
            gold[i, arm] /= len(rows)
    no_cot = {r["item_id"]: r for r in read_json_rows(INPUTS[1])}
    q = {
        i: sum(a == items[i]["gold"] for a in no_cot[i]["sampled_answers"]) / 10
        for i in items
    }
    pao = {}
    for i in ids:
        target = draws[i]["corrupted"][0]["propagation_answer"]
        pao[i] = sum(a == target for a in no_cot[i]["sampled_answers"]) / 10
    return Data(draws, items, ids, trip, dom, lvl, prop, gold, q, pao, by_draw_id)


# ---------------------------------------------------------------- statistics
def rnd(x: float, n: int = 2) -> float:
    return round(float(x), n)


def group_by_triplet(d: Data, sel: list[str], vals: dict[str, float]) -> dict:
    by: dict[str, list[float]] = defaultdict(list)
    for i in sel:
        by[d.trip[i]].append(vals[i])
    return by


def boot_ci(by: dict[str, list[float]]) -> list[float]:
    """Whole-triplet percentile bootstrap of the pooled mean, in percentage points."""
    ts = sorted(by)
    rng = random.Random(SEED)
    stats = []
    for _ in range(REPLICATES):
        picked = [v for t in (rng.choice(ts) for _ in ts) for v in by[t]]
        stats.append(float(np.mean(picked)))
    stats.sort()
    return [rnd(100 * stats[249], 1), rnd(100 * stats[9749], 1)]


def cde_values(d: Data) -> dict[str, float]:
    return {i: d.prop[i, "corrupted"] - d.prop[i, "placebo"] for i in d.ids}


def cde_block(d: Data, sel: list[str]) -> dict[str, Any]:
    vals = cde_values(d)
    return {
        "items": len(sel),
        "triplets": len({d.trip[i] for i in sel}),
        "cde": rnd(100 * np.mean([vals[i] for i in sel])),
        "ci95": boot_ci(group_by_triplet(d, sel, vals)),
    }


def protocol_complete(d: Data) -> list[str]:
    return [
        i
        for i in d.ids
        if all(r["protocol_valid"] is True for a in d.draws[i] for r in d.draws[i][a])
    ]


def coverage(d: Data) -> dict[str, Any]:
    rows = [r for i in d.ids for a in d.draws[i] for r in d.draws[i][a]]
    invalid = Counter(r["arm"] for r in rows if r["protocol_valid"] is not True)
    from vnedu_cot.language_checks import classify_reasoning_language

    pc = protocol_complete(d)
    full = sum(c == 3 for c in Counter(d.trip[i] for i in pc).values())
    texts = [t for r in rows if (t := (r.get("reasoning_continuation") or "").strip())]
    language = Counter(classify_reasoning_language(t).value for t in texts)
    return {
        "nonempty_continuations": len(texts),
        "continuation_language": dict(language),
        "draws": len(rows),
        "unparseable": sum(r["parsed_answer"] not in list("ABCD") for r in rows),
        "protocol_invalid_by_arm": dict(invalid),
        "protocol_invalid_total": sum(invalid.values()),
        "protocol_complete_items": len(pc),
        "protocol_complete_triplets": full,
        "levels": dict(Counter(d.lvl.values())),
    }


def table1(d: Data) -> dict[str, Any]:
    out = {}
    vals = cde_values(d)
    for name in (*DOMAINS, "overall"):
        sel = [i for i in d.ids if name == "overall" or d.dom[i] == name]
        mean = {a: np.mean([d.prop[i, a] for i in sel]) for a in ARMS}
        out[name] = {
            "n_items": len(sel),
            "placebo_pct": rnd(100 * mean["placebo"]),
            "corrupted_pct": rnd(100 * mean["corrupted"]),
            "cde": rnd(100 * np.mean([vals[i] for i in sel])),
            "ci95": boot_ci(group_by_triplet(d, sel, vals)),
            "nongold_corrupted_pct": rnd(
                100 * (1 - np.mean([d.gold[i, "corrupted"] for i in sel]))
            ),
        }
    return out


def edit_and_distribution(d: Data) -> dict[str, Any]:
    edit = {i: d.prop[i, "placebo"] - d.prop[i, "clean"] for i in d.ids}
    rows = {a: [r for i in d.ids for r in d.draws[i][a]] for a in ARMS}
    counts = [d.prop[i, "corrupted"] * 20 for i in d.ids]
    return {
        "propagation_counts": {
            a: sum(r["parsed_answer"] == r["propagation_answer"] for r in rows[a])
            for a in ARMS
        },
        "edit_contrast": {
            "estimate": rnd(100 * np.mean(list(edit.values()))),
            "ci95": boot_ci(group_by_triplet(d, d.ids, edit)),
        },
        "nongold_pct": {
            a: rnd(
                100
                * sum(r["parsed_answer"] != r["gold_answer"] for r in rows[a])
                / len(rows[a]),
                1,
            )
            for a in ARMS
        },
        "item_distribution_corrupted": {
            "zero": sum(x == 0 for x in counts),
            "partial": sum(0 < x < 20 for x in counts),
            "full": sum(x == 20 for x in counts),
        },
    }


def hits(d: Data, i: str) -> np.ndarray:
    rows = d.draws[i]["corrupted"]
    return np.array([int(r["parsed_answer"] == r["propagation_answer"]) for r in rows])


def clustering(d: Data) -> dict[str, Any]:
    y = {i: hits(d, i) for i in d.ids}
    k, n = 20, len(d.ids)
    grand = np.mean([v for i in d.ids for v in y[i]])
    msb = k * sum((np.mean(y[i]) - grand) ** 2 for i in d.ids) / (n - 1)
    msw = sum((v - np.mean(y[i])) ** 2 for i in d.ids for v in y[i]) / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    tm: dict[str, list[float]] = defaultdict(list)
    for i in d.ids:
        tm[d.trip[i]].append(float(np.mean(y[i])))
    gm2 = np.mean([np.mean(y[i]) for i in d.ids])
    nt, m = len(tm), 3
    msb2 = m * sum((np.mean(v) - gm2) ** 2 for v in tm.values()) / (nt - 1)
    msw2 = sum((x - np.mean(v)) ** 2 for v in tm.values() for x in v) / (nt * (m - 1))
    total = np.concatenate([y[i] for i in d.ids]).var()
    within = np.mean([y[i].var() for i in d.ids])
    item_means = np.array([y[i].mean() for i in d.ids])
    vt = np.var([np.mean(v) for v in tm.values()])
    return {
        "anova_icc_item": rnd(icc, 3),
        "design_effect": rnd(1 + (k - 1) * icc, 1),
        "anova_icc_triplet_item_means": rnd((msb2 - msw2) / (msb2 + (m - 1) * msw2), 3),
        "variance_share": {
            "total": rnd(total, 4),
            "within_item": rnd(within, 4),
            "icc_item": rnd(1 - within / total, 3),
            "triplet_share": rnd(vt / total, 3),
            "icc_triplet_among_item_means": rnd(vt / np.var(item_means), 3),
        },
    }


def contrasts(d: Data) -> dict[str, Any]:
    vals = cde_values(d)
    by: dict[str, dict[str, float]] = defaultdict(dict)
    for i in d.ids:
        by[d.trip[i]][d.dom[i]] = vals[i]
    out = {}
    for a, b in [
        ("computation", "history_literature"),
        ("theoretical", "history_literature"),
    ]:
        per_trip = {t: [v[a] - v[b]] for t, v in by.items()}
        out[f"{a}_minus_{b}"] = {
            "estimate": rnd(100 * np.mean([v[0] for v in per_trip.values()]), 1),
            "ci95": boot_ci(per_trip),
        }
    return out


def schemas(d: Data) -> dict[str, Any]:
    out = {}
    for sc in (f"S{k}" for k in range(1, 7)):
        sel = [i for i in d.ids if i.startswith(sc + "-")]
        block = cde_block(d, sel)
        block["corrupted_propagation_draws"] = int(
            sum(d.prop[i, "corrupted"] * 20 for i in sel)
        )
        block["corrupted_draws"] = 20 * len(sel)
        out[sc] = block
    return out


# ---------------------------------------------------------------- regression
def t_cdf(t: float, df: int) -> float:
    x = np.linspace(0, abs(t), 20001)
    dens = (1 + x**2 / df) ** (-(df + 1) / 2)
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return float(0.5 + np.trapezoid(dens, x) * c * np.sign(t))


def cr1_fit(y: np.ndarray, cl: np.ndarray, cols: dict[str, np.ndarray]):
    x = np.column_stack([np.ones(len(y)), *cols.values()])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    groups = set(cl)
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    meat = sum(
        np.outer(x[cl == g].T @ resid[cl == g], x[cl == g].T @ resid[cl == g])
        for g in groups
    )
    g_n = len(groups)
    cov = g_n / (g_n - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    table = {}
    for name, b, s in zip(["const", *cols], beta, se):
        t = b / s
        p = 2 * (1 - t_cdf(abs(t), g_n - 1))
        table[name] = {"coef": rnd(b), "se": rnd(s), "t": rnd(t), "p": rnd(p, 3)}
    return table, beta, cov, g_n


def regressions(d: Data) -> dict[str, Any]:
    ids = d.ids
    vals = cde_values(d)
    y = np.array([100 * vals[i] for i in ids])
    cl = np.array([d.trip[i] for i in ids])
    q = np.array([d.q[i] for i in ids])
    flag = {n: np.array([d.dom[i] == n for i in ids], float) for n in DOMAINS}
    th, hi, co = (flag[n] for n in ("theoretical", "history_literature", "computation"))
    ext = np.array([d.items[i]["n_external_premises"] > 0 for i in ids], float)
    pao = np.array([d.pao[i] for i in ids])
    out: dict[str, Any] = {}
    out["q_domain"] = cr1_fit(y, cl, {"q": q, "theo": th, "his": hi})[0]
    out["q_domain_distractor"] = cr1_fit(
        y, cl, {"q": q, "pao": pao, "theo": th, "his": hi}
    )[0]
    table, beta, cov, g_n = cr1_fit(
        y, cl, {"q": q, "comp": co, "theo": th, "qXcomp": q * co, "qXtheo": q * th}
    )
    sub = cov[4:6, 4:6]
    wald = float(beta[4:6] @ np.linalg.inv(sub) @ beta[4:6])
    table["omnibus_wald"] = {
        "chi2_2": rnd(wald),
        "p": rnd(math.exp(-wald / 2), 3),
        "F_2_G-1": rnd(wald / 2),
        "G": g_n,
    }
    out["interaction_history_reference"] = table
    for name in ("computation", "theoretical"):
        s = flag[name] == 1
        out[f"within_{name}"] = cr1_fit(y[s], cl[s], {"q": q[s]})[0]
    out["external_load"] = cr1_fit(y, cl, {"q": q, "theo": th, "his": hi, "ext": ext})[
        0
    ]
    return out


def difficulty(d: Data) -> dict[str, Any]:
    dropped = [i for i in d.items if i not in d.dom]
    out = {
        "q_executed_pct": rnd(100 * np.mean([d.q[i] for i in d.ids]), 1),
        "q_dropped_pct": rnd(100 * np.mean([d.q[i] for i in dropped]), 1),
        "q_all_pct": rnd(100 * np.mean(list(d.q.values())), 1),
        "q_by_domain_pct": {
            n: rnd(100 * np.mean([d.q[i] for i in d.ids if d.dom[i] == n]), 1)
            for n in DOMAINS
        },
    }
    comp = [i for i in d.ids if d.dom[i] == "computation"]
    ext = {}
    for label, sel in [
        (
            "with_external_premises",
            [i for i in comp if d.items[i]["n_external_premises"] > 0],
        ),
        (
            "no_external_premises",
            [i for i in comp if d.items[i]["n_external_premises"] == 0],
        ),
    ]:
        corr = {i: d.prop[i, "corrupted"] for i in sel}
        ext[label] = {
            "items": len(sel),
            "corrupted_pct": rnd(100 * np.mean(list(corr.values())), 1),
            "ci95": boot_ci(group_by_triplet(d, sel, corr)),
        }
    out["computation_external_load"] = ext
    return out


def behaviour(d: Data) -> dict[str, Any]:
    from vnedu_cot.language_checks import contains_error_language

    def cont(r: dict[str, Any]) -> str:
        return r.get("reasoning_continuation") or ""

    corr = [(i, r) for i in d.ids for r in d.draws[i]["corrupted"]]
    gold = [(i, r) for i, r in corr if r["parsed_answer"] == r["gold_answer"]]
    prop = [(i, r) for i, r in corr if r["parsed_answer"] == r["propagation_answer"]]
    explicit = sum(bool(REPAIR_RE.search(cont(r))) for _, r in gold)
    doubt = sum(bool(CONFLICT_RE.search(cont(r))) for _, r in prop)
    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, r in gold:
        hit = bool(REPAIR_RE.search(cont(r)))
        key = "easy" if d.q[i] >= 0.85 else "hard" if d.q[i] < 0.6 else "mid"
        strata[key][0] += 1
        strata[key][1] += not hit
        if d.q[i] < 0.6 and d.dom[i] != "history_literature":
            strata["hardquant"][0] += 1
            strata["hardquant"][1] += hit
    thresholds = []
    for label, test in [
        ("q<0.6", lambda x: x < 0.6),
        ("q<=0.6", lambda x: x <= 0.6),
        ("q<0.65", lambda x: x < 0.65),
    ]:
        for quant in (False, True):
            sel = [
                r
                for i, r in gold
                if test(d.q[i]) and not (quant and d.dom[i] == "history_literature")
            ]
            h = sum(bool(REPAIR_RE.search(cont(r))) for r in sel)
            thresholds.append(
                {
                    "rule": label,
                    "quant_only": quant,
                    "gold": len(sel),
                    "explicit": h,
                    "pct": rnd(100 * h / len(sel), 1),
                }
            )
    hard = [r for i, r in gold if d.q[i] < 0.6]
    fields = {
        f: sum(bool(REPAIR_RE.search(cont(r) + " " + (r.get(f) or ""))) for r in hard)
        for f in ("full_reasoning", "raw_final_response")
    }
    pkg_gold = sum(contains_error_language(cont(r)) for _, r in gold)
    return {
        "gold_returns": len(gold),
        "gold_explicit_repair": explicit,
        "gold_explicit_pct": rnd(100 * explicit / len(gold), 1),
        "gold_silent": len(gold) - explicit,
        "gold_silent_pct": rnd(100 * (len(gold) - explicit) / len(gold), 1),
        "propagation_returns": len(prop),
        "propagation_conflict": doubt,
        "propagation_no_doubt": len(prop) - doubt,
        "propagation_no_doubt_pct": rnd(100 * (len(prop) - doubt) / len(prop), 1),
        "strata_n_gold_and_silent_hardquant_explicit": dict(strata),
        "keyword_thresholds": thresholds,
        "hard_items_explicit_with_extra_field": fields,
        "package_classifier_gold": {
            "flagged": pkg_gold,
            "of": len(gold),
            "pct": rnd(100 * pkg_gold / len(gold), 1),
        },
        "package_classifier_propagation": {
            "flagged": sum(contains_error_language(cont(r)) for _, r in prop),
            "of": len(prop),
        },
        "non_empty_continuations": sum(
            bool(cont(r).strip())
            for i in d.ids
            for a in d.draws[i]
            for r in d.draws[i][a]
        ),
    }


# ---------------------------------------------------------------- probes
def probe_pass(d: Data) -> tuple[dict[str, bool], dict[str, list[bool]]]:
    probes: dict[str, list[bool]] = defaultdict(list)
    for r in read_json_rows(INPUTS[2]):
        if "r1-0528" in r.get("model", ""):
            probes[r["item_id"]].append(r["correct"])
    return {i: all(probes[i]) if probes.get(i) else True for i in d.items}, probes


def probe_block(d: Data, ok: dict[str, bool], probes: dict) -> dict[str, Any]:
    probed = [i for i in d.ids if probes.get(i)]
    vals = cde_values(d)
    out: dict[str, Any] = {
        "knowledge_ok": sum(ok[i] for i in d.ids),
        "of": len(d.ids),
        "probed_items": len(probed),
        "probed_passed": sum(ok[i] for i in probed),
    }
    zero_load = [i for i in d.ids if not probes.get(i)]
    for label, sel in [
        ("pass", [i for i in d.ids if ok[i]]),
        ("fail", [i for i in d.ids if not ok[i]]),
        ("probed_pass", [i for i in probed if ok[i]]),
        ("probed_all", probed),
        ("zero_load", zero_load),
    ]:
        out[f"cde_probe_{label}"] = {
            "items": len(sel),
            "cde": rnd(100 * np.mean([vals[i] for i in sel])),
            "ci95": boot_ci(group_by_triplet(d, sel, vals)),
        }
    return out


# ---------------------------------------------------------------- agreement
def qwk(a: list[int], b: list[int], k: int) -> float:
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return float(1 - (w * obs).sum() / (w * exp).sum())


def cohen(a: list[bool], b: list[bool]) -> float:
    x, y = np.array(a), np.array(b)
    po = np.mean(x == y)
    pe = np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y))
    return float((po - pe) / (1 - pe))


def ac1(a: list[int], b: list[int], k: int) -> float:
    po = np.mean([x == y for x, y in zip(a, b)])
    shares = [(a.count(c) + b.count(c)) / (2 * len(a)) for c in range(k)]
    pe = sum(p * (1 - p) for p in shares) / (k - 1)
    return float((po - pe) / (1 - pe))


def load_job_a() -> tuple[dict, dict, dict]:
    """Return corrupted ratings, catch ratings, placebo confirmations per item."""
    mapping = {}
    for r in read_json_rows(INPUTS[4]):
        mapping[r["rating_id"]] = (r["item_id"], r["is_catch_trial"])
    placebo_key = {r["review_id"]: r["item_id"] for r in read_json_rows(INPUTS[5])}
    rated: dict[str, dict] = {}
    catch: dict[str, dict] = {}
    placebo: dict[str, dict] = defaultdict(dict)
    for who in ("rater1", "rater2"):
        for x in sheet_rows(RATED / f"rating_sheet_deepseek_{who}_rated.xlsx", 2):
            if x[0] not in mapping:
                continue
            item, is_catch = mapping[x[0]]
            rel = str(x[7]).upper()
            rel_val = None if rel in ("NONE", "N/A", "") else rel == "TRUE"
            target = catch if is_catch else rated
            target.setdefault(item, {})[who] = (int(x[6] or 0), rel_val)
        for x in sheet_rows(RATED / f"placebo_review_deepseek_{who}_rated.xlsx", 2):
            if x[0] in placebo_key:
                placebo[placebo_key[x[0]]][who] = str(x[4]).upper() == "YES"
    return rated, catch, placebo


def screen_pass(rated: dict, placebo: dict, i: str) -> bool:
    ja, pl = rated.get(i, {}), placebo.get(i, {})
    both_a = len(ja) == 2 and all(det > 0 and rel is True for det, rel in ja.values())
    return both_a and len(pl) == 2 and all(pl.values())


def human_ratings(d: Data, rated: dict, catch: dict, placebo: dict) -> dict[str, Any]:
    its = [i for i in rated if i in d.dom]
    a = [rated[i]["rater1"][0] for i in its]
    b = [rated[i]["rater2"][0] for i in its]
    vals = cde_values(d)

    def rel_pairs(sel: list[str]) -> list[tuple[bool, bool]]:
        pairs = [(rated[i]["rater1"][1], rated[i]["rater2"][1]) for i in sel]
        return [(x, y) for x, y in pairs if x is not None and y is not None]

    per_domain = {}
    for n in DOMAINS:
        s = [i for i in its if d.dom[i] == n]
        pairs = rel_pairs(s)
        multi = len({v for p in pairs for v in p}) > 1
        per_domain[n] = {
            "detectability_qwk": rnd(
                qwk(
                    [rated[i]["rater1"][0] for i in s],
                    [rated[i]["rater2"][0] for i in s],
                    6,
                ),
                3,
            ),
            "relevance_kappa": rnd(cohen(*map(list, zip(*pairs))), 3)
            if multi
            else None,
            "severity_counts": [
                sum(rated[i][r][0] == k for i in s for r in rated[i]) for k in range(6)
            ],
            "detectability_mean": rnd(
                np.mean([rated[i][r][0] for i in s for r in rated[i]])
            ),
            "detectability_sd": rnd(
                np.std([rated[i][r][0] for i in s for r in rated[i]], ddof=1)
            ),
        }
    rv = rel_pairs(its)
    pp = sum(x and y for x, y in rv)
    nn = sum((not x) and (not y) for x, y in rv)
    po = (pp + nn) / len(rv)
    pos = sum(x + y for x, y in rv) / (2 * len(rv))
    pe = 2 * pos * (1 - pos)
    pv = [(v["rater1"], v["rater2"]) for v in placebo.values()]
    allj = [rated[i][r][0] for i in its for r in rated[i]]
    means = {i: float(np.mean([rated[i][r][0] for r in rated[i]])) for i in its}
    return {
        "corrupted_items": len(rated),
        "catch_stimuli": len(catch),
        "catch_judgments": sum(len(v) for v in catch.values()),
        "catch_false_alarms": sum(
            dt > 0 for v in catch.values() for dt, _ in v.values()
        ),
        "detectability": {
            "qwk": rnd(qwk(a, b, 6), 3),
            "exact": sum(x == y for x, y in zip(a, b)),
            "diff_ge_2": sum(abs(x - y) >= 2 for x, y in zip(a, b)),
            "within_1": sum(abs(x - y) <= 1 for x, y in zip(a, b)),
            "rater1_mean": rnd(np.mean(a)),
            "rater2_mean": rnd(np.mean(b)),
            "rater1_higher": sum(x > y for x, y in zip(a, b)),
            "rater2_higher": sum(y > x for x, y in zip(a, b)),
            "rater_pearson_r": rnd(np.corrcoef(a, b)[0, 1], 2),
            "mean": rnd(np.mean(allj)),
            "sd": rnd(np.std(allj, ddof=1)),
            "item_mean_mean": rnd(np.mean(list(means.values()))),
            "item_mean_sd": rnd(np.std(list(means.values()), ddof=1)),
            "r_with_cde": rnd(
                np.corrcoef([means[i] for i in its], [vals[i] for i in its])[0, 1], 3
            ),
        },
        "relevance": {
            "valid": len(rv),
            "both_pos": pp,
            "both_neg": nn,
            "split": len(rv) - pp - nn,
            "observed_agreement_pct": rnd(100 * po),
            "positive_rate_pct": rnd(100 * pos, 1),
            "kappa": rnd(cohen(*map(list, zip(*rv))), 3),
            "pabak": rnd(2 * po - 1, 3),
            "ac1": rnd((po - pe) / (1 - pe), 3),
        },
        "placebo": {
            "n": len(pv),
            "both_yes": sum(x and y for x, y in pv),
            "both_no": sum(not x and not y for x, y in pv),
            "split": sum(x != y for x, y in pv),
            "observed_agreement_pct": rnd(100 * np.mean([x == y for x, y in pv])),
            "kappa": rnd(cohen(*map(list, zip(*pv))), 3),
        },
        "by_domain": per_domain,
    }


def disclosure(d: Data) -> dict[str, Any]:
    from vnedu_cot.language_checks import contains_error_language

    key = {r["review_id"]: r for r in read_json_rows(INPUTS[6])}
    sheets = {}
    for who in ("rater1", "rater2", "llm"):
        rows = sheet_rows(RATED / f"disclosure_rating_sheet_{who}_rated.xlsx", 1)
        sheets[who] = {r[0]: int(r[6]) for r in rows if r[0] and r[0] != "review_id"}
    ids = sorted(set(sheets["rater1"]) & set(sheets["rater2"]))
    cor = [i for i in ids if key[i]["arm"] != "clean"]
    human_a = [sheets["rater1"][i] for i in ids]
    human_b = [sheets["rater2"][i] for i in ids]
    llm = [sheets["llm"][i] for i in cor]
    out: dict[str, Any] = {
        "n": len(ids),
        "n_corrupted": len(cor),
        "human_qwk": rnd(qwk(human_a, human_b, 4), 3),
        "human_exact": sum(x == y for x, y in zip(human_a, human_b)),
        "answer_mix_corrupted": dict(
            Counter(
                "gold"
                if key[i]["model_parsed_answer"] == key[i]["gold_answer"]
                else "prop"
                if key[i]["model_parsed_answer"] == key[i]["propagation_answer"]
                else "other"
                for i in cor
            )
        ),
        "llm_vs_human": {},
        "lexical": {},
    }
    for who in ("rater2", "rater1"):
        h = [sheets[who][i] for i in cor]
        out["llm_vs_human"][who] = {
            "exact_pct": rnd(100 * np.mean([x == y for x, y in zip(llm, h)]), 1),
            "qwk": rnd(qwk(llm, h, 4), 3),
            "ac1": rnd(ac1(llm, h, 4), 3),
            "capture": sum(x > 0 and y > 0 for x, y in zip(llm, h)),
            "human_positive": sum(y > 0 for y in h),
            "llm_positive_human_zero": sum(x > 0 and y == 0 for x, y in zip(llm, h)),
        }
    text = {
        i: d.by_draw_id[key[i]["draw_id"]].get("reasoning_continuation") or ""
        for i in cor
    }
    for name, flagger in [
        ("keyword", lambda s: bool(REPAIR_RE.search(s))),
        ("package_contains_error_language", contains_error_language),
    ]:
        flags = {i: bool(flagger(text[i])) for i in cor}
        block: dict[str, Any] = {"flags": sum(flags.values()), "of": len(cor)}
        for who in ("rater1", "rater2"):
            pos = {i: sheets[who][i] > 0 for i in cor}
            tp = sum(flags[i] and pos[i] for i in cor)
            block[who] = {
                "positives": sum(pos.values()),
                "precision": rnd(tp / max(1, sum(flags.values()))),
                "recall": rnd(tp / sum(pos.values())),
                "false_positives": sum(flags[i] and not pos[i] for i in cor),
            }
        out["lexical"][name] = block
    return out


# ---------------------------------------------------------------- sensitivity
def sensitivity(d: Data, screen: dict[str, bool], ok: dict[str, bool]) -> dict:
    pc = set(protocol_complete(d))
    hs = [i for i in d.ids if screen[i]]
    strict = [i for i in hs if i in pc and ok[i]]
    count = Counter(d.trip[i] for i in strict)
    complete = [i for i in strict if count[d.trip[i]] == 3]
    out = {
        "human_screened": hs,
        "human_screened_protocol_probe": strict,
        "human_screened_complete_triplets": complete,
        "protocol_complete": sorted(pc),
        "excluding_S3-PRI-02-PHY": [i for i in d.ids if i != "S3-PRI-02-PHY"],
    }
    return {k: cde_block(d, v) for k, v in out.items()}


# ---------------------------------------------------------------- output
def write_csv(d: Data, screen: dict, ok: dict) -> None:
    complete = set(protocol_complete(d))
    cols = [
        "item_id", "triplet_id", "domain", "level", "arm", "draws", "propagation",
        "gold", "unparseable", "protocol_invalid", "all_arms_protocol_valid",
        "human_screen_pass", "probe_pass", "q_i",
    ]  # fmt: skip
    with (OUT_DIR / "item_outcomes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        for i in d.ids:
            for arm in ARMS:
                rows = d.draws[i][arm]
                prop = sum(r["parsed_answer"] == r["propagation_answer"] for r in rows)
                gold = sum(r["parsed_answer"] == r["gold_answer"] for r in rows)
                bad = sum(r["parsed_answer"] not in list("ABCD") for r in rows)
                inv = sum(r["protocol_valid"] is not True for r in rows)
                writer.writerow(
                    [i, d.trip[i], d.dom[i], d.lvl[i], arm, len(rows), prop, gold]
                    + [bad, inv, i in complete, screen[i], ok[i], d.q[i]]
                )


def summarise(stats: dict[str, Any]) -> None:
    t1 = stats["table1"]
    print("Paper statistics")
    print(f"  draws {stats['coverage']['draws']}, items {t1['overall']['n_items']}")
    for name, row in t1.items():
        print(f"  CDE {name:20s} {row['cde']:6.2f}  CI {row['ci95']}")
    print(f"  edit contrast {stats['outcomes']['edit_contrast']}")
    print(f"  item ICC {stats['clustering']['anova_icc_item']}")
    for name, row in stats["sensitivity"].items():
        print(f"  {name}: n={row['items']} CDE={row['cde']} CI={row['ci95']}")


def main() -> None:
    sys.path.insert(0, "src")
    d = load_data()
    rated, catch, placebo = load_job_a()
    ok, probes = probe_pass(d)
    screen = {i: screen_pass(rated, placebo, i) for i in d.ids}
    stats: dict[str, Any] = {
        "inputs": {str(p): sha256(p) for p in INPUTS},
        "method": {
            "bootstrap": "whole-triplet percentile, 2.5/97.5",
            "bootstrap_seed": SEED,
            "bootstrap_replicates": REPLICATES,
            "cluster": "triplet",
            "p_values": "t(G-1) with CR1 cluster-robust covariance",
            "unparseable": "kept in denominators as Y=0",
        },
        "coverage": coverage(d),
        "table1": table1(d),
        "outcomes": edit_and_distribution(d),
        "clustering": clustering(d),
        "contrasts": contrasts(d),
        "schemas": schemas(d),
        "protocol_complete_cde": cde_block(d, protocol_complete(d)),
        "difficulty": difficulty(d),
        "regressions": regressions(d),
        "behaviour": behaviour(d),
        "probes": probe_block(d, ok, probes),
        "human_ratings": human_ratings(d, rated, catch, placebo),
        "disclosure": disclosure(d),
        "sensitivity": sensitivity(d, screen, ok),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_csv(d, screen, ok)
    summarise(stats)


if __name__ == "__main__":
    main()
