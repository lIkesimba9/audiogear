#!/usr/bin/env python
"""Per-dataset QA report -> a single self-contained HTML file.

For every dataset with an ``extended_metadata.csv`` this collects:
  - a summary (clips, hours, speakers, % kept by the current filter thresholds);
  - the drop-reason breakdown (same ``reasons_to_drop``/``TH`` as
    ``filter_clean.py`` — one source of truth, the report cannot drift from the
    actual filter);
  - the distribution of every filtered metric (SVG histogram with the threshold
    drawn in, plus the failing share and the missing/sentinel share);
  - LISTENABLE samples right at each threshold, from both sides (base64
    ``<audio>``) — the calibration tool: if clips "just below" sound fine the
    threshold is too strict, if clips "just above" sound bad it is too loose;
  - random accepted/rejected clips.

Usage:
    export AUDIOGEAR_DATA_DIR=/path/to/data_root
    python examples/qa_report.py                  # all datasets
    python examples/qa_report.py resd dialogs     # only the named ones
    python examples/qa_report.py resd --out /tmp/resd.html --samples 4

Default output: <AUDIOGEAR_DATA_DIR>/<ds>/qa_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import os
import random
import sys

import filter_clean as fc  # sibling module: TH / HUMAN_TEXT / ASR_TEXT / reasons_to_drop

csv.field_size_limit(10**9)

MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac"}


def metric_specs(ds):
    """(column, label, threshold kind, threshold). Kind: ge / le / range."""
    s = [
        ("distillmos", "MOS (DistillMOS)", "ge", fc.TH["distillmos"]),
        ("wada_snr", "SNR (WADA), dB", "ge", fc.TH["wada_snr"]),
        ("pyt_stoi", "STOI", "ge", fc.TH["stoi"]),
        ("duration", "Duration, s", "range", (fc.TH["dur_min"], fc.TH["dur_max"])),
        ("bandwidth_hz", "Bandwidth, Hz", "ge", fc.TH["bandwidth_min"]),
        ("speaking_rate", "Speaking rate, phonemes/s", "range", (fc.TH["srate_min"], fc.TH["srate_max"])),
    ]
    if ds in fc.HUMAN_TEXT:
        s.append(("whisper_cer", "CER (whisper)", "le", fc.TH["cer_max"]))
    if ds in fc.ASR_TEXT:
        s.append(("asr_agreement", "ASR agreement", "ge", fc.TH["asr_agree_min"]))
    return s


def val(r, col):
    """float or None; negative metric values are sentinels ("could not score")."""
    v = fc.fnum(r.get(col))
    if v is None or math.isnan(v):
        return None
    if v < 0:  # every reported metric has a non-negative valid range
        return None
    return v


def passes(kind, thr, v):
    if kind == "ge":
        return v >= thr
    if kind == "le":
        return v <= thr
    lo, hi = thr
    return lo <= v <= hi


def _pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def svg_hist(vals, kind, thr, width=640, height=150, bins=48):
    """Histogram with the failing side tinted and the threshold line(s) drawn. Pure SVG."""
    lo, hi = _pct(vals, 0.005), _pct(vals, 0.995)
    if isinstance(thr, tuple):
        lo, hi = min(lo, thr[0]), max(hi, thr[1])
    else:
        lo, hi = min(lo, thr), max(hi, thr)
    if hi <= lo:
        hi = lo + 1.0
    counts = [0] * bins
    for v in vals:
        i = int((min(max(v, lo), hi) - lo) / (hi - lo) * (bins - 1))
        counts[i] += 1
    peak = max(counts) or 1
    bw = width / bins
    parts = [f'<svg viewBox="0 0 {width} {height + 18}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;max-width:{width}px">']
    for i, c in enumerate(counts):
        center = lo + (i + 0.5) / bins * (hi - lo)
        ok = passes(kind, thr, center)
        bh = c / peak * (height - 8)
        parts.append(f'<rect x="{i * bw:.1f}" y="{height - bh:.1f}" width="{bw - 1:.1f}" '
                     f'height="{bh:.1f}" fill="{"#7da7d9" if ok else "#e08a8a"}"/>')
    thrs = thr if isinstance(thr, tuple) else (thr,)
    for t in thrs:
        x = (t - lo) / (hi - lo) * width
        parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#c02020" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{min(x + 3, width - 40):.1f}" y="11" font-size="11" fill="#c02020">{t:g}</text>')
    parts.append(f'<text x="0" y="{height + 14}" font-size="11" fill="#777">{lo:.3g}</text>')
    parts.append(f'<text x="{width}" y="{height + 14}" font-size="11" fill="#777" text-anchor="end">{hi:.3g}</text>')
    parts.append("</svg>")
    return "".join(parts)


class AudioEmbedder:
    """base64 embedding with budgets: per clip and for the whole report."""

    def __init__(self, root, clip_mb, total_mb):
        self.root, self.clip_cap, self.budget = root, clip_mb * 2**20, total_mb * 2**20
        self.missing = 0

    def embed(self, audio_file):
        for p in (audio_file, os.path.join(self.root, audio_file)):
            if os.path.isfile(p):
                size = os.path.getsize(p)
                if size > self.clip_cap or size > self.budget:
                    return None  # too big — the caller tries the next candidate
                self.budget -= size
                mime = MIME.get(os.path.splitext(p)[1].lower(), "audio/wav")
                b64 = base64.b64encode(open(p, "rb").read()).decode()
                return f'<audio controls preload="none" src="data:{mime};base64,{b64}"></audio>'
        self.missing += 1
        return None


def sample_card(r, emb, note=""):
    audio = emb.embed(r.get("audio_file") or "")
    if audio is None:
        return None
    text = html.escape((r.get("text") or "").strip()[:140])
    meta = (f'id={html.escape(str(r.get("id")))} · {fc.fnum(r.get("duration")) or 0:.1f} s'
            + (f' · {html.escape(note)}' if note else ""))
    return (f'<div class="card">{audio}<div class="cap"><b>{meta}</b>'
            f'<br>mos={r.get("distillmos", "")} snr={r.get("wada_snr", "")} stoi={r.get("pyt_stoi", "")}'
            f'<br><i>{text}</i></div></div>')


def near_threshold(rows, col, kind, thr, n):
    """n clips closest to the threshold on each side: (failing, passing)."""
    scored = [(r, val(r, col)) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    thrs = thr if isinstance(thr, tuple) else (thr,)
    dist = lambda v: min(abs(v - t) for t in thrs)
    failing = sorted(((r, v) for r, v in scored if not passes(kind, thr, v)), key=lambda x: dist(x[1]))
    passing = sorted(((r, v) for r, v in scored if passes(kind, thr, v)), key=lambda x: dist(x[1]))
    return failing[: n * 4], passing[: n * 4]  # extra candidates — some may fail to embed


def render_samples(pairs, col, emb, n):
    cards = []
    for r, v in pairs:
        card = sample_card(r, emb, note=f"{col}={v:g}")
        if card:
            cards.append(card)
        if len(cards) >= n:
            break
    return "\n".join(cards) or "<p class='dim'>no samples (audio missing or embed budget exhausted)</p>"


CSS = """
body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1080px;padding:0 16px;color:#222;background:#fff}
h1{font-size:22px} h2{font-size:17px;margin-top:28px} .dim{color:#888}
.cards{display:flex;flex-wrap:wrap;gap:10px} .card{border:1px solid #ddd;border-radius:8px;padding:8px;width:315px}
.card audio{width:100%} .cap{font-size:12px;margin-top:4px;line-height:1.35}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0}
.stat{border:1px solid #ddd;border-radius:8px;padding:8px 14px} .stat b{font-size:18px;display:block}
table{border-collapse:collapse;font-size:13px} td,th{border:1px solid #ddd;padding:4px 10px;text-align:left}
details{margin:14px 0;border:1px solid #e5e5e5;border-radius:8px;padding:8px 14px}
summary{cursor:pointer;font-weight:600;font-size:15px}
@media (prefers-color-scheme:dark){body{background:#181818;color:#ddd}
.card,.stat,details{border-color:#3a3a3a} td,th{border-color:#3a3a3a} .dim{color:#999}}
"""


def build_report(ds, rows, data_root, n_samples, clip_mb, total_mb):
    rng = random.Random(42)
    emb = AudioEmbedder(os.path.join(data_root, ds), clip_mb, total_mb)
    from collections import Counter

    reasons = Counter()
    keep, reject = [], []
    for r in rows:
        bad = fc.reasons_to_drop(ds, r)
        (reject if bad else keep).append(r)
        reasons.update(bad)

    hours = sum(fc.fnum(r.get("duration")) or 0 for r in rows) / 3600
    speakers = {(r.get("speaker_id") or r.get("speaker") or "").strip() for r in rows} - {""}

    out = [f"<title>QA {html.escape(ds)}</title><style>{CSS}</style>",
           f"<h1>QA report: {html.escape(ds)}</h1>",
           '<div class="stats">',
           f'<div class="stat"><b>{len(rows):,}</b>clips</div>',
           f'<div class="stat"><b>{hours:.1f}</b>hours</div>',
           f'<div class="stat"><b>{len(speakers)}</b>speakers</div>',
           f'<div class="stat"><b>{100 * len(keep) // max(1, len(rows))}%</b>kept (filter_clean)</div>',
           "</div>"]

    out.append("<h2>Drop reasons</h2><table><tr><th>reason</th><th>clips</th><th>% of all</th></tr>")
    for reason, cnt in reasons.most_common():
        out.append(f"<tr><td>{reason}</td><td>{cnt:,}</td><td>{100 * cnt / max(1, len(rows)):.1f}%</td></tr>")
    out.append("</table>")

    out.append("<h2>Metrics and thresholds</h2>"
               "<p class='dim'>The dashed red line is the filter_clean threshold. Listen to the samples right at "
               "the threshold: if the rejected ones sound fine the threshold is too strict, if the accepted ones "
               "sound bad it is too loose.</p>")
    for col, label, kind, thr in metric_specs(ds):
        vals = [v for v in (val(r, col) for r in rows) if v is not None]
        n_missing = len(rows) - len(vals)
        if not vals:
            out.append(f"<details><summary>{label} — no data ({col})</summary></details>")
            continue
        n_fail = sum(1 for v in vals if not passes(kind, thr, v))
        thr_txt = f"[{thr[0]:g}, {thr[1]:g}]" if isinstance(thr, tuple) else f"{thr:g}"
        out.append(f"<details open><summary>{label} · threshold {thr_txt} · failing "
                   f"{100 * n_fail / len(vals):.1f}% · missing {100 * n_missing / len(rows):.1f}%</summary>")
        out.append(svg_hist(vals, kind, thr))
        failing, passing = near_threshold(rows, col, kind, thr, n_samples)
        out.append(f"<p><b>Rejected, right at the threshold</b> ({col}):</p><div class='cards'>"
                   + render_samples(failing, col, emb, n_samples) + "</div>")
        out.append(f"<p><b>Accepted, right at the threshold</b> ({col}):</p><div class='cards'>"
                   + render_samples(passing, col, emb, n_samples) + "</div></details>")

    out.append("<h2>Random samples</h2>")
    for title, pool in (("Accepted", keep), ("Rejected", reject)):
        picks = rng.sample(pool, min(n_samples, len(pool))) if pool else []
        cards = []
        for r in picks:
            note = " ".join(fc.reasons_to_drop(ds, r)) if title == "Rejected" else ""
            card = sample_card(r, emb, note=note)
            if card:
                cards.append(card)
        out.append(f"<h3>{title} ({len(pool):,})</h3><div class='cards'>"
                   + ("\n".join(cards) or "<p class='dim'>no samples</p>") + "</div>")

    if emb.missing:
        out.append(f"<p class='dim'>⚠ audio not found for {emb.missing} sample candidate(s)</p>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="dataset names (empty = all with extended_metadata.csv)")
    ap.add_argument("--out", default=None, help="output html path (single dataset only)")
    ap.add_argument("--samples", type=int, default=5, help="samples per group")
    ap.add_argument("--clip-mb", type=float, default=2.0, help="per-file embed size limit")
    ap.add_argument("--total-mb", type=float, default=60.0, help="total audio budget for the report")
    args = ap.parse_args()

    data_root = fc.DATA_ROOT
    dss = args.datasets or sorted(
        d for d in os.listdir(data_root)
        if os.path.isfile(os.path.join(data_root, d, "extended_metadata.csv")))
    if args.out and len(dss) != 1:
        sys.exit("--out works with a single dataset only")
    for ds in dss:
        ext = os.path.join(data_root, ds, "extended_metadata.csv")
        if not os.path.isfile(ext):
            print(f"[skip] {ds}: no extended_metadata.csv")
            continue
        with open(ext, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="|"))
        if not rows:
            print(f"[skip] {ds}: empty")
            continue
        html_doc = build_report(ds, rows, data_root, args.samples, args.clip_mb, args.total_mb)
        out = args.out or os.path.join(data_root, ds, "qa_report.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"[{ds}] {len(rows):,} rows -> {out} ({os.path.getsize(out) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
