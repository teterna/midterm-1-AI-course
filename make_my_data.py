"""Midterm 1 data generator: CSS4007 Artificial Intelligence, Narxoz University.

    python make_my_data.py --github teterna --code ZEBRA-28

Writes your personal exam data to ./my_data/ (two CSV files and DATA.md). The same
username and code always give byte-identical files, on any machine and any Python >= 3.9.
Standard library only. Rehearse with `--code PRACTICE`; the real code is announced in class.

Every number in the output is synthetic: nothing here is a measurement of a real system.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

GENERATOR_VERSION = "1"
LANGS = ("EN", "RU", "KK")
CONDITIONS = ("clean", "degraded")
CANDIDATES = (
    ("A", 3),
    ("A", 5),
    ("B", 3),
    ("B", 5),
)  # (chunking config, retrieved chunks k)
N_QUESTIONS = 60
N_DOCS_PER_CELL = 30
Z95 = 1.96
MAX_ATTEMPTS = 20000
E5_PER_256_CL100K = {
    "EN": 263,
    "RU": 128,
    "KK": 76,
}  # Lecture 4.1, slide 23 (mean chunk length)

_MASK = (1 << 64) - 1
_USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_CODE_RE = re.compile(r"^[A-Za-z0-9-]{3,32}$")


# --------------------------------------------------------------------------------------
# Deterministic randomness (integer arithmetic only, so no platform or version drift)
# --------------------------------------------------------------------------------------
class Rng:
    """SplitMix64 stream seeded from a string."""

    def __init__(self, material: str) -> None:
        self.material = material
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        self._state = int.from_bytes(digest[:8], "big")

    def child(self, label: str) -> Rng:
        """Independent sub-stream, so editing one part never shifts another."""
        return Rng(self.material + "/" + label)

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        return (self._next() >> 11) / float(1 << 53)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def randint(self, a: int, b: int) -> int:
        return a + self._next() % (b - a + 1)

    def choice(self, seq: Sequence[Any]) -> Any:
        return seq[self._next() % len(seq)]

    def shuffle(self, items: list[Any]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]


# --------------------------------------------------------------------------------------
# Statistics shared with the grading helper
# --------------------------------------------------------------------------------------
def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion hits/n."""
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def annual_cost(p: dict[str, Any], cfg: str, k: int) -> float:
    """Annual input+output cost in USD of retrieving k chunks of config `cfg` per request."""
    per_request = 0.0
    for lang in LANGS:
        tokens_in = (
            p["system_tokens"]
            + p["question_tokens"][lang]
            + k * p["chunk_tokens"][cfg][lang]
        )
        tokens_out = p["answer_tokens"][lang]
        cost = (tokens_in * p["price_in"] + tokens_out * p["price_out"]) / 1e6
        per_request += p["share"][lang] / 100.0 * cost
    return per_request * p["requests_per_day"] * 365


def decide(
    cands: list[dict[str, Any]], budget: float
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Best affordable candidate, the pool statistically tied with it, and the cheapest of the pool.

    Two recalls are treated as not distinguishable when their 95 % Wilson intervals overlap.
    """
    affordable = [c for c in cands if c["cost"] <= budget]
    best = max(affordable, key=lambda c: (c["hits"], -c["cost"]))
    pool = [c for c in affordable if c["lo"] <= best["hi"] and best["lo"] <= c["hi"]]
    return best, pool, min(pool, key=lambda c: c["cost"])


# --------------------------------------------------------------------------------------
# Shared inputs
# --------------------------------------------------------------------------------------
def _draw_premiums(r: Rng) -> dict[str, float]:
    """Tokens per word-equivalent relative to English for Model M's tokenizer."""
    return {
        "EN": 1.0,
        "RU": round(r.uniform(1.25, 2.3), 2),
        "KK": round(r.uniform(1.8, 3.8), 2),
    }


# --------------------------------------------------------------------------------------
# Part 1: retrieval and cost
# --------------------------------------------------------------------------------------
def _draw_part1_inputs(r: Rng, prem: dict[str, float]) -> dict[str, Any]:
    en_a, en_b = r.randint(240, 275), r.randint(62, 80)
    q_en, a_en = r.randint(24, 44), r.randint(90, 150)
    kk, ru = r.randint(15, 40), r.randint(25, 45)
    while 100 - kk - ru < 15:
        ru -= 1
    price_in = r.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    price_out = price_in * r.choice([5, 6])
    return {
        "chunk_tokens": {
            "A": {lang: round(en_a * prem[lang]) for lang in LANGS},
            "B": {lang: round(en_b * prem[lang]) for lang in LANGS},
        },
        "question_tokens": {lang: round(q_en * prem[lang]) for lang in LANGS},
        "answer_tokens": {lang: round(a_en * prem[lang]) for lang in LANGS},
        "share": {"EN": 100 - kk - ru, "RU": ru, "KK": kk},
        "system_tokens": r.randint(180, 320),
        "requests_per_day": 500 * r.randint(12, 28),
        "price_in": price_in,
        "price_out": price_out,
    }


def _budget(p: dict[str, Any], cut: int) -> int:
    """Budget between the cheapest-but-`cut` excluded candidates and the affordable ones."""
    costs = sorted(annual_cost(p, cfg, k) for cfg, k in CANDIDATES)
    lo, hi = costs[len(costs) - 1 - cut], costs[len(costs) - cut]
    budget = int(round((lo + hi) / 2.0, -2))
    if not lo <= budget < hi:
        raise RuntimeError("budget does not separate candidates")
    return budget


def _targets(r: Rng, cfg: str) -> tuple[float, float, float, float]:
    if cfg == "A":
        t1 = r.uniform(0.45, 0.65)
        t3 = t1 + r.uniform(0.10, 0.18)
        t5 = t3 + r.uniform(0.05, 0.10)
        t10 = t5 + r.uniform(0.03, 0.07)
    else:
        t1 = r.uniform(0.25, 0.45)
        t3 = t1 + r.uniform(0.15, 0.25)
        t5 = t3 + r.uniform(0.12, 0.20)
        t10 = t5 + r.uniform(0.04, 0.08)
    return t1, min(t3, 0.97), min(t5, 0.98), min(t10, 0.99)


def _bucket(r: Rng, x: float, t: tuple[float, float, float, float]) -> int:
    if x < t[0]:
        return 1
    if x < t[1]:
        return r.randint(2, 3)
    if x < t[2]:
        return r.randint(4, 5)
    if x < t[3]:
        return r.randint(6, 10)
    return 11


_SHIFT = {
    "EN": -0.04,
    "RU": 0.0,
    "KK": 0.06,
}  # Kazakh questions are harder (Lecture 4.1)


def _draw_ranks(r: Rng) -> list[tuple[str, str, int, int]]:
    langs = [lang for lang in LANGS for _ in range(N_QUESTIONS // len(LANGS))]
    r.shuffle(langs)
    ta, tb = _targets(r, "A"), _targets(r, "B")
    rows = []
    for i, lang in enumerate(langs):
        d = r.random() + _SHIFT[lang]
        rank_a = _bucket(r, d + r.uniform(-0.12, 0.12), ta)
        rank_b = _bucket(r, d + r.uniform(-0.12, 0.12), tb)
        rows.append((f"q{i + 1:02d}", lang, rank_a, rank_b))
    return rows


def candidate_table(
    rows: Sequence[tuple[str, str, int, int]], p: dict[str, Any]
) -> list[dict[str, Any]]:
    cands = []
    for cfg, k in CANDIDATES:
        idx = 2 if cfg == "A" else 3
        hits = sum(1 for row in rows if row[idx] <= k)
        lo, hi = wilson(hits, len(rows))
        cands.append(
            {
                "cfg": cfg,
                "k": k,
                "hits": hits,
                "lo": lo,
                "hi": hi,
                "cost": annual_cost(p, cfg, k),
            }
        )
    return cands


def _accept(
    situation: str,
    best: dict[str, Any],
    pool: list[dict[str, Any]],
    choice: dict[str, Any],
) -> bool:
    if situation == "clear":
        return len(pool) == 1
    return len(pool) >= 2 and choice is not best and best["hits"] - choice["hits"] <= 5


def _draw_part1(r: Rng, prem: dict[str, float]) -> dict[str, Any]:
    p = _draw_part1_inputs(r.child("inputs"), prem)
    situation = r.choice(["clear", "tie"])
    cut = r.choice([1, 2])
    p["budget"] = _budget(p, cut)
    rr = r.child("ranks")
    for attempt in range(MAX_ATTEMPTS):
        rows = _draw_ranks(rr.child(str(attempt)))
        best, pool, choice = decide(candidate_table(rows, p), p["budget"])
        if _accept(situation, best, pool, choice):
            p.update(rows=rows, situation=situation, cut=cut, attempts=attempt + 1)
            return p
    raise RuntimeError("no ranking satisfied the situation constraints")


# --------------------------------------------------------------------------------------
# Part 2: document pipelines
# --------------------------------------------------------------------------------------
def _draw_part2(r: Rng) -> dict[str, Any]:
    base = {"H": r.uniform(0.955, 0.985), "L": r.uniform(0.915, 0.945)}
    degrade = {"H": r.uniform(0.05, 0.12), "L": r.uniform(0.08, 0.16)}
    lang_pen = {
        "H": {"EN": 0.0, "RU": r.uniform(0.01, 0.04), "KK": r.uniform(0.03, 0.08)},
        "L": {"EN": 0.0, "RU": r.uniform(0.02, 0.06), "KK": r.uniform(0.04, 0.10)},
    }
    weak_lang = "KK" if r.random() < 0.7 else "RU"
    weak_cond = "degraded" if r.random() < 0.65 else "clean"
    hurt = r.choice(["H", "L"])
    extra = {
        hurt: r.uniform(0.10, 0.20),
        ("L" if hurt == "H" else "H"): r.uniform(0.02, 0.05),
    }
    results = []
    for pipe in ("H", "L"):
        for lang in LANGS:
            for cond in CONDITIONS:
                acc = base[pipe] - lang_pen[pipe][lang] + r.uniform(-0.03, 0.03)
                if cond == "degraded":
                    acc -= degrade[pipe]
                if (lang, cond) == (weak_lang, weak_cond):
                    acc -= extra[pipe]
                acc = min(0.995, max(0.20, acc))
                results.append(
                    (
                        pipe,
                        lang,
                        cond,
                        N_DOCS_PER_CELL,
                        min(N_DOCS_PER_CELL, round(N_DOCS_PER_CELL * acc)),
                    )
                )
    width, height = r.choice(
        [(827, 1169), (1240, 1754), (1654, 2339)]
    )  # A4 at 100 / 150 / 200 dpi
    price_in = r.choice([1.0, 2.0, 3.0])
    return {
        "results": results,
        "weak_cell": (weak_lang, weak_cond),
        "hurt": hurt,
        "page_px": (width, height),
        "image_tokens": math.ceil(width / 28) * math.ceil(height / 28),
        "out_tokens_per_page": r.randint(280, 420),
        "price_in": price_in,
        "price_out": price_in * 5,
        "pages_per_day": 500 * r.randint(6, 18),
        "pii_share": r.randint(8, 35),
        "server_month": r.choice([450, 600, 750, 900]),
        "server_capacity": r.choice([4000, 6000, 8000]),
    }


# --------------------------------------------------------------------------------------
# Part 3: five short questions, each tied to the student's own numbers
# --------------------------------------------------------------------------------------
def _q1(r: Rng, prem: dict[str, float]) -> dict[str, Any]:
    template = r.choice(["a", "b"])
    if template == "a":
        chars_en = r.randint(285, 320)
        chars = {
            "EN": chars_en,
            "RU": round(chars_en * r.uniform(1.00, 1.12)),
            "KK": round(chars_en * r.uniform(1.04, 1.18)),
        }
        bytes_ = {
            "EN": chars_en,
            "RU": round(chars["RU"] * r.uniform(1.86, 1.96)),
            "KK": round(chars["KK"] * r.uniform(1.88, 2.02)),
        }
        tok_en = round(chars_en / r.uniform(5.0, 5.6))
        tokens = {
            "EN": tok_en,
            "RU": round(tok_en * prem["RU"]),
            "KK": round(tok_en * prem["KK"]),
        }
        text = (
            "One complaint sentence, measured three ways (synthetic numbers):\n\n"
            "| | characters | UTF-8 bytes | tokens (Model M) |\n|---|---|---|---|\n"
            + "\n".join(
                f"| {lang} | {chars[lang]} | {bytes_[lang]} | {tokens[lang]} |"
                for lang in LANGS
            )
            + "\n\nCompute the KK÷EN ratio for characters, for bytes and for tokens. "
            "**Number:** the token ratio divided by the byte ratio (2 decimals). "
            "**Why** (at most 40 words): what does this number say about where the Kazakh premium "
            "comes from, the encoding or the tokenizer?"
        )
        return {
            "template": "a",
            "chars": chars,
            "bytes": bytes_,
            "tokens": tokens,
            "text": text,
        }
    in_en, out_en = r.randint(1100, 1800), r.randint(100, 170)
    in_kk, out_kk = (
        round(in_en * r.uniform(2.2, 3.2)),
        round(out_en * r.uniform(1.4, 2.0)),
    )
    p_in = r.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    p_out = p_in * r.choice([5, 6])
    text = (
        f"The same support request, once in English and once in Kazakh, through Model M "
        f"(${p_in:g} per million input tokens, ${p_out:g} per million output tokens):\n\n"
        "| | input tokens | output tokens |\n|---|---|---|\n"
        f"| EN | {in_en:,} | {out_en:,} |\n| KK | {in_kk:,} | {out_kk:,} |\n\n"
        "Compute the token ratio (total tokens KK ÷ EN) and the bill ratio (cost KK ÷ cost EN). "
        "**Number:** the bill ratio (2 decimals). "
        "**Why** (at most 40 words): why is the bill ratio not equal to the token ratio?"
    )
    return {
        "template": "b",
        "in_en": in_en,
        "out_en": out_en,
        "in_kk": in_kk,
        "out_kk": out_kk,
        "p_in": p_in,
        "p_out": p_out,
        "text": text,
    }


def _q2(r: Rng) -> dict[str, Any]:
    template = r.choice(["a", "b"])
    size = r.choice([512, 576, 640, 768, 896])
    intro = (
        "A teammate sets the chunk size to "
        f"**{size} tokens, counted with cl100k**. The retrieval model (e5-small) reads at most 512 of "
        "*its own* tokens. Assume e5 tokens grow in proportion to cl100k tokens at the rate measured in "
        "Lecture 4.1 (a chunker told 256 cl100k tokens produced chunks of 263 / 128 / 76 e5 tokens for "
        "EN / RU / KK), and that every chunk is exactly the full size.\n\n"
    )
    if template == "a":
        text = intro + (
            "**Number:** the share of an English chunk, in percent (1 decimal), that the embedding never sees. "
            "**Why** (at most 40 words): which of the three languages are cut, and what happens to the cut part?"
        )
        return {"template": "a", "size": size, "text": text}
    frac = r.choice([55, 60, 65, 70, 75, 80, 85, 90, 95])
    text = intro + (
        f"In an English chunk, the sentence that contains the answer starts {frac} % of the way into the chunk "
        "and is shorter than 20 e5 tokens. "
        "**Number:** the position of the start of that sentence, in e5 tokens (nearest integer). "
        "**Why** (at most 40 words): can a question that matches only that sentence retrieve this chunk, and would you see an error?"
    )
    return {"template": "b", "size": size, "frac": frac, "text": text}


def _draw_logits(r: Rng) -> list[float]:
    top = round(r.uniform(3.5, 5.0), 1)
    logits = [top]
    for _ in range(4):
        logits.append(round(logits[-1] - r.uniform(0.3, 1.5), 1))
    return logits


def _q3(r: Rng) -> dict[str, Any]:
    template = r.choice(["a", "b"])
    words = ["Monday", "Friday", "Sunday", "holiday", "Easter"]
    for _ in range(MAX_ATTEMPTS):
        logits = _draw_logits(r)
        if len(set(logits)) < 5:
            continue
        table = "| token | score |\n|---|---|\n" + "\n".join(
            f"| {w} | {s:.1f} |" for w, s in zip(words, logits)
        )
        head = (
            'A language model gives these next-token scores for the prompt "The branch is closed on":\n\n'
            + table
            + "\n\n"
        )
        if template == "a":
            temp = r.choice([0.4, 0.5, 0.7, 1.5, 2.0])
            top_p = softmax(logits, temp)[0]
            if not 0.25 < top_p < 0.95:
                continue
            text = head + (
                f"Apply softmax at temperature T = {temp:g}. **Number:** the probability of the top token, "
                "in percent (1 decimal). **Why** (at most 40 words): what did changing T do to the ranking of the tokens, and to the set of tokens that can be sampled?"
            )
            return {
                "template": "a",
                "words": words,
                "logits": logits,
                "temperature": temp,
                "text": text,
            }
        nucleus = r.choice([0.8, 0.9, 0.95])
        probs = softmax(logits)
        cum, kept, ok = 0.0, 0, True
        for pr in probs:
            cum += pr
            kept += 1
            if abs(cum - nucleus) < 1e-6:
                ok = False
            if cum >= nucleus:
                break
        if not ok or not 2 <= kept <= 4:
            continue
        text = head + (
            f"Use temperature T = 1, then apply top-p = {nucleus:g}. **Number:** the probability of the top token "
            "after top-p and renormalisation, in percent (1 decimal). **Why** (at most 40 words): how does top-p differ from lowering the temperature?"
        )
        return {
            "template": "b",
            "words": words,
            "logits": logits,
            "top_p": nucleus,
            "text": text,
        }
    raise RuntimeError("no usable logits")


def _q4(r: Rng) -> dict[str, Any]:
    template = r.choice(["a", "b"])
    if template == "a":
        c_x = r.randint(66, 76)
        a_y = r.randint(12, 24)
        c_y = c_x - r.randint(2, 6)
        penalty = r.choice([1, 2])
        text = (
            "Two models answer the same 100 factual questions (1 point per correct answer on the public leaderboard):\n\n"
            '| | correct | wrong | "I don\'t know" |\n|---|---|---|---|\n'
            f"| Model X | {c_x} | {100 - c_x} | 0 |\n| Model Y | {c_y} | {100 - c_y - a_y} | {a_y} |\n\n"
            f'Now score them again with a penalty: +1 per correct answer, −{penalty} per wrong answer, 0 for "I don\'t know". '
            "**Number:** Y's penalised score minus X's (negative if X is ahead). "
            "**Why** (at most 40 words): which model does the public leaderboard prefer, and what does that teach a model to do?"
        )
        return {
            "template": "a",
            "c_x": c_x,
            "c_y": c_y,
            "a_y": a_y,
            "penalty": penalty,
            "text": text,
        }
    sets = [
        ("Birthdays of public figures", r.randint(200, 400), r.randint(15, 35), True),
        (
            "Account numbers in synthetic bank profiles",
            r.randint(100, 300),
            r.randint(20, 40),
            True,
        ),
        ("Two-digit multiplication", r.randint(100, 200), r.randint(1, 10), False),
        (
            "Continuing arithmetic sequences",
            r.randint(150, 200),
            r.randint(1, 5),
            False,
        ),
    ]
    r.shuffle(sets)
    rows = "\n".join(f"| {name} | {n} | {pct} % |" for name, n, pct, _ in sets)
    text = (
        "A base model (pre-training only) is evaluated on four question sets. For each set, the table gives how many "
        "questions it has and the share of its facts that appear exactly once in the training corpus:\n\n"
        "| set | questions | facts seen exactly once |\n|---|---|---|\n" + rows + "\n\n"
        "**Number:** the lower bound on the expected share of wrong answers, in percent (1 decimal), over all questions "
        'in the sets where such a bound applies. **Why** (at most 40 words): which sets did you leave out and why, and would a system-prompt instruction "never invent a fact" change the bound?'
    )
    return {
        "template": "b",
        "sets": [(name, n, pct, arbitrary) for name, n, pct, arbitrary in sets],
        "text": text,
    }


def _q5(r: Rng) -> dict[str, Any]:
    template = r.choice(["a", "b"])
    if template == "a":
        params_b = r.choice([7, 13, 34, 70, 180])
        tpp = r.randint(5, 400)
        tokens_b = params_b * tpp
        shown = (
            f"{tokens_b / 1000:g} trillion"
            if tokens_b >= 1000
            else f"{tokens_b:,} billion"
        )
        text = (
            f"A lab trains a model with **{params_b} billion parameters on {shown} tokens**. "
            "Chinchilla (Lecture 1.1) trained 70 billion parameters on 1.4 trillion tokens, which the paper presents as the "
            "compute-optimal ratio. **Number:** this model's tokens per parameter divided by Chinchilla's tokens per parameter (2 decimals). "
            "**Why** (at most 40 words): is the model under- or over-trained by that criterion, and what did Chinchilla say about models such as GPT-3?"
        )
        return {
            "template": "a",
            "params_b": params_b,
            "tokens_b": tokens_b,
            "text": text,
        }
    per_turn, turns = r.randint(60, 150), r.randint(12, 30)
    text = (
        f"A support chat adds **{per_turn} new tokens per turn** (customer message plus assistant reply) and has **{turns} turns**. "
        "The model keeps no state, so the full history is sent again on every turn (Lecture 1.2). "
        "As in the Lecture 1.2 table (100, 200, 300, ... tokens billed on turns 1, 2, 3, ...), turn *i* is billed "
        "for the tokens of all *i* turns so far, including its own. "
        "**Number:** the total input tokens billed over the whole conversation. "
        "**Why** (at most 40 words): how many times has the first message been paid for by the last turn, and what does prompt caching change?"
    )
    return {"template": "b", "per_turn": per_turn, "turns": turns, "text": text}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
class Bundle(NamedTuple):
    params: dict[str, Any]
    files: dict[str, str]
    fingerprint: str


def _csv(header: str, rows: Sequence[Sequence[Any]]) -> str:
    return (
        header + "\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
    )


def _render_part1(p: dict[str, Any]) -> str:
    table = "| language | share of traffic | question tokens | answer tokens | chunk A tokens | chunk B tokens |\n|---|---|---|---|---|---|\n"
    table += "\n".join(
        f"| {lang} | {p['share'][lang]} % | {p['question_tokens'][lang]} | {p['answer_tokens'][lang]} | "
        f"{p['chunk_tokens']['A'][lang]} | {p['chunk_tokens']['B'][lang]} |"
        for lang in LANGS
    )
    return (
        "## Part 1: inputs\n\n"
        f"`retrieval_ranks.csv`: {N_QUESTIONS} test questions ({N_QUESTIONS // 3} EN, {N_QUESTIONS // 3} RU, {N_QUESTIONS // 3} KK). "
        "`rank_A` and `rank_B` give the position of the correct chunk in the top 10 retrieved under config A and config B; "
        "**11 means the correct chunk was not in the top 10.**\n\n"
        "Production traffic and Model M token counts (tokens per item, as counted by Model M's tokenizer):\n\n"
        f"{table}\n\n"
        f"- Requests per day: **{p['requests_per_day']:,}**, in the traffic shares above.\n"
        f"- System prompt: **{p['system_tokens']}** tokens in every request.\n"
        f"- Price: **${p['price_in']:g}** per million input tokens, **${p['price_out']:g}** per million output tokens.\n"
        f"- Annual budget for this service (model calls only): **${p['budget']:,}**.\n"
    )


def _render_part2(q: dict[str, Any]) -> str:
    width, height = q["page_px"]
    return (
        "## Part 2: inputs\n\n"
        "`doc_results.csv`: for each pipeline, language and scan condition, `n_docs` test invoices and `n_correct` "
        "= invoices whose number, date, seller ID and total were all extracted exactly (strict scorer, as in Lab 04).\n\n"
        "- **Pipeline H**: hosted vision-language API in a data centre abroad, pay per token; each page is sent as an image.\n"
        "- **Pipeline L**: open vision-language model on your own server in Kazakhstan, fixed monthly cost.\n\n"
        f"- Volume: **{q['pages_per_day']:,}** scanned pages per day. In **{q['pii_share']} %** of them an individual's name and ИИН are printed "
        "(sole proprietors, receipts).\n"
        f"- Page image: **{width} × {height}** pixels. Pipeline H counts an image as ⌈width ÷ 28⌉ × ⌈height ÷ 28⌉ input tokens and returns "
        f"about **{q['out_tokens_per_page']}** output tokens per page (the JSON).\n"
        f"- Pipeline H price: **${q['price_in']:g}** per million input tokens, **${q['price_out']:g}** per million output tokens.\n"
        f"- Pipeline L: one server costs **${q['server_month']:,}** per month and handles up to **{q['server_capacity']:,}** pages per day; "
        "add servers as needed.\n"
    )


def generate(username: str, code: str) -> Bundle:
    """Build all exam files for one student. Raises ValueError on a malformed username or code."""
    if not _USERNAME_RE.match(username):
        raise ValueError("not a valid GitHub username: " + repr(username))
    if not _CODE_RE.match(code):
        raise ValueError(
            "the session code must be 3-32 letters, digits or '-': " + repr(code)
        )
    material = f"midterm1|v{GENERATOR_VERSION}|{username.lower()}|{code.lower()}"
    root = Rng(material)
    prem = _draw_premiums(root.child("tokenizer"))
    p1 = _draw_part1(root.child("part1"), prem)
    p2 = _draw_part2(root.child("part2"))
    p3 = {
        "q1": _q1(root.child("q1"), prem),
        "q2": _q2(root.child("q2")),
        "q3": _q3(root.child("q3")),
        "q4": _q4(root.child("q4")),
        "q5": _q5(root.child("q5")),
    }
    ranks_csv = _csv("qid,lang,rank_A,rank_B", p1["rows"])
    docs_csv = _csv("pipeline,lang,condition,n_docs,n_correct", p2["results"])
    body = (
        _render_part1(p1)
        + "\n"
        + _render_part2(p2)
        + "\n## Part 3: your five questions\n\nAnswer each in `ANSWERS.md`: one **Number** line and one **Why** line of at most 40 words.\n\n"
        + "\n\n".join(
            f"### {key.upper()}\n\n{p3[key]['text']}"
            for key in ("q1", "q2", "q3", "q4", "q5")
        )
        + "\n"
    )
    digest = hashlib.sha256()
    for chunk in (ranks_csv, docs_csv, body):
        digest.update(chunk.encode("utf-8"))
    fingerprint = digest.hexdigest()[:8]
    header = (
        "# Midterm 1: your data\n\n"
        f"GitHub username: `{username.lower()}` · session code: `{code.lower()}` · generator v{GENERATOR_VERSION}\n\n"
        f"**Fingerprint: `{fingerprint}`**: copy it into the first lines of `ANSWERS.md`. "
        "All numbers below are synthetic: this is an exam, not a measurement.\n\n"
    )
    params = {
        "part1": p1,
        "part2": p2,
        "part3": p3,
        "premiums": prem,
        "username": username.lower(),
        "code": code.lower(),
    }
    files = {
        "retrieval_ranks.csv": ranks_csv,
        "doc_results.csv": docs_csv,
        "DATA.md": header + body,
    }
    return Bundle(params, files, fingerprint)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate your personal Midterm 1 data."
    )
    parser.add_argument(
        "--github",
        required=True,
        help="your GitHub username (the one that owns your fork)",
    )
    parser.add_argument(
        "--code",
        required=True,
        help="session code announced in class (PRACTICE for rehearsal)",
    )
    parser.add_argument(
        "--out", default="my_data", help="output folder (default: my_data)"
    )
    args = parser.parse_args(argv)
    try:
        bundle = generate(args.github, args.code)
    except ValueError as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in bundle.files.items():
        (out / name).write_bytes(text.encode("utf-8"))
    print(
        f"Wrote {len(bundle.files)} files to {out}/ · fingerprint {bundle.fingerprint}"
    )
    print("Next: open my_data/DATA.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
