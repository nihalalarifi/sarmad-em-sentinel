"""
Slide-ready figures (Arabic labels) from eval/metrics.json and eval/live_run.json.

    python make_figures.py        -> eval/slides/*.png  (dark and light versions)
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(t):
        return get_display(arabic_reshaper.reshape(t))
except ImportError:          # pip install arabic-reshaper python-bidi
    def ar(t):
        return t

OUT = Path("eval/slides")
THEMES = {
    "dark": dict(bg="#0A1620", ink="#E4EEF0", muted="#8FA7B0", grid="#1E3A48",
                 sig="#3FE0C5", alert="#FF6B5E", amber="#F6B94C", grey="#5B6F78"),
    "light": dict(bg="#FFFFFF", ink="#0E2A33", muted="#4F6A73", grid="#D5E0E2",
                  sig="#0E9C88", alert="#D9473B", amber="#B7791F", grey="#9AAAB0"),
}
SCN_AR = {"idle": "خمول", "active": "استخدام نشط", "background": "خدمات خلفية", "boot": "إقلاع",
          "interference": "تداخل بيئي", "covert_fast": "قناة خفية سريعة", "covert_stealth": "قناة خفية متخفية"}


def style(ax, c):
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=c["muted"], labelsize=13, length=0)
    ax.grid(axis="y", color=c["grid"], lw=1)
    ax.set_axisbelow(True)


def fig_compare(m, c, name):
    r = m["per_snr"]["10"]
    dr_ai = r["tp_count"] / r["n_attack"]
    vals = {"ai": [dr_ai * 100, r["fpr"] * 100], "en": [r["energy_naive"]["dr"] * 100, r["energy_naive"]["fpr"] * 100]}
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=c["bg"])
    style(ax, c)
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, vals["ai"], w, color=c["sig"], label=ar("سرمد (ذكاء اصطناعي)"))
    b2 = ax.bar(x + w / 2, vals["en"], w, color=c["grey"], label=ar("عتبة الطاقة التقليدية"))
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%" if v >= 1 else f"{v:.2f}%",
                    ha="center", color=c["ink"], fontsize=17, fontweight="bold")
    ax.set_xticks(x, [ar("معدل الاكتشاف (أعلى أفضل)"), ar("معدل الإنذار الكاذب (أقل أفضل)")], fontsize=15, color=c["ink"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, fontsize=14, labelcolor=c["ink"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, 0.015, ar(f"بيانات اختبار مستقلة: {r['n_attack']} نافذة هجوم و{r['n_normal']} نافذة طبيعية") + "   (SNR 10 dB)",
             ha="center", color=c["muted"], fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / name, dpi=150, facecolor=c["bg"])
    plt.close(fig)


def fig_snr(m, c, name):
    snrs = [0, 5, 10, 15, 20]
    P = m["per_snr"]
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=c["bg"])
    style(ax, c)
    ax.axvspan(-1.5, 2.5, color=c["alert"], alpha=0.10, lw=0)
    ax.text(0.5, 6, ar("خارج نطاق التدريب") + "\n" + ar("يلزم إعادة المعايرة"), ha="center", color=c["alert"], fontsize=12)
    ax.plot(snrs, [P[str(s)]["dr_fast"] * 100 for s in snrs], "o-", color=c["sig"], lw=3, ms=9, label=ar("اكتشاف القناة السريعة"))
    ax.plot(snrs, [P[str(s)]["dr_stealth"] * 100 for s in snrs], "o-", color=c["amber"], lw=3, ms=9, label=ar("اكتشاف القناة المتخفية"))
    ax.plot(snrs, [P[str(s)]["fpr"] * 100 for s in snrs], "o--", color=c["alert"], lw=2.5, ms=8, label=ar("الإنذار الكاذب"))
    ax.set_xticks(snrs, [f"{s} dB" for s in snrs])
    ax.set_xlim(-1.5, 21)
    ax.set_ylim(-3, 106)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(ar("نسبة الإشارة إلى الضوضاء (أعلى = إشارة أوضح)"), color=c["muted"], fontsize=13)
    ax.legend(frameon=False, fontsize=14, labelcolor=c["ink"], loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=c["bg"])
    plt.close(fig)


def fig_scenarios(m, c, name):
    S = m["per_scenario_10db"]
    keys = ["idle", "active", "background", "boot", "interference", "covert_fast", "covert_stealth"]
    vals = [S[k]["flagged"] * 100 for k in keys]
    cols = [c["grey"]] * 5 + [c["alert"]] * 2
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=c["bg"])
    style(ax, c)
    bars = ax.bar(range(len(keys)), vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", color=c["ink"], fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(keys)), [ar(SCN_AR[k]) for k in keys], fontsize=13, color=c["ink"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.text(2, 70, ar("نشاط طبيعي: لا يُنبَّه عليه"), ha="center", color=c["muted"], fontsize=14)
    ax.text(5.5, 108, ar("هجوم"), ha="center", color=c["alert"], fontsize=14)
    fig.text(0.5, 0.015, ar("نسبة النوافذ التي تجاوزت حد الشذوذ لكل سيناريو") + "   (SNR 10 dB)", ha="center", color=c["muted"], fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=150, facecolor=c["bg"])
    plt.close(fig)


def fig_timeline(c, name):
    """Live run: score over time, threshold, and when alerts fired (persistence at work)."""
    lr = json.loads(Path("eval/live_run.json").read_text())
    if not (Path(lr["truth_file"]).exists() and Path(lr["decisions_file"]).exists()):
        print("  (live-run logs not found, keeping the existing timeline figure)")
        return
    truth = {int(r["seq"]): r for r in csv.DictReader(open(lr["truth_file"]))}
    dec = [r for r in csv.DictReader(open(lr["decisions_file"]))]
    scen = [truth.get(int(r["seq"]), {}).get("scenario", "") for r in dec]
    attack = np.array([x in ("covert_fast", "covert_stealth") for x in scen])
    starts = [i for i in range(100, len(attack)) if attack[i] and not attack[i - 100:i].any()]
    if not starts:
        return
    s0 = starts[0]
    e0 = s0
    while e0 < len(attack) and attack[e0]:
        e0 += 1
    start, end = s0 - 100, min(len(dec), e0 + 25)
    sc = np.array([float(r["score"]) for r in dec])[start:end]
    st = [r["status"] for r in dec][start:end]
    al = [bool(r["alert_id"].strip()) for r in dec][start:end]
    sn = scen[start:end]
    t = np.arange(len(sc))
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=c["bg"])
    style(ax, c)
    labels = {"idle": ("خمول", None), "active": ("حمل ثقيل", c["amber"]), "covert_fast": ("قناة خفية", c["alert"]),
              "covert_stealth": ("قناة خفية", c["alert"]), "background": ("خدمات خلفية", None), "boot": ("إقلاع", None),
              "interference": ("تداخل", None)}
    i = 0
    while i < len(sn):
        j = i
        while j < len(sn) and sn[j] == sn[i]:
            j += 1
        lab, col = labels.get(sn[i], (sn[i], None))
        if col:
            ax.axvspan(i - 0.5, j - 0.5, color=col, alpha=0.12, lw=0)
        ax.text((i + j - 1) / 2, 1.04, ar(lab), ha="center", color=col or c["muted"], fontsize=14)
        i = j
    ax.plot(t, sc, color=c["sig"], lw=2)
    ax.axhline(0.70, color=c["alert"], ls="--", lw=1.6)
    ax.text(2, 0.73, ar("حد الشذوذ 0.70"), ha="left", color=c["alert"], fontsize=12)
    w = [k for k, s_ in enumerate(st) if s_ == "warning"]
    ax.scatter(w, sc[w], s=45, color=c["amber"], zorder=3, label=ar("تجاوز مؤقت، لم يُرسل تنبيه"))
    a = [k for k, v in enumerate(al) if v]
    ax.scatter(a, sc[a], s=190, marker="v", color=c["alert"], zorder=4, label=ar("أُرسل تنبيه للوحة"))
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel(ar("النوافذ المتتالية في تشغيل حي عبر النظام الكامل"), color=c["muted"], fontsize=13)
    ax.set_ylabel(ar("درجة الشذوذ"), color=c["muted"], fontsize=13)
    ax.legend(frameon=False, fontsize=13, labelcolor=c["ink"], loc="center left", bbox_to_anchor=(0.0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=c["bg"])
    plt.close(fig)


def fig_summary(m, c, name):
    """One slide: three key numbers + AI vs classic comparison."""
    r = m["per_snr"]["10"]
    ev = m["event_10db"]
    dr = r["tp_count"] / r["n_attack"]
    fig = plt.figure(figsize=(16, 9), facecolor=c["bg"])
    fig.text(0.965, 0.90, ar("نتائج قياس الأداء"), ha="right", color=c["ink"], fontsize=30, fontweight="bold")
    fig.text(0.965, 0.855, ar("بيانات اختبار مستقلة لم يرها النموذج أثناء التدريب"), ha="right", color=c["muted"], fontsize=15)

    # key numbers (right column): metric names in English, explanation in Arabic
    kpis = [
        (f"{dr*100:.1f}%", "Detection Rate", "نسبة حالات الهجوم التي اكتشفها سرمد",
         f"{r['tp_count']} من {r['n_attack']} نافذة هجوم"),
        (f"{r['fpr']*100:.2f}%", "False Positive Rate", "نسبة الحالات الطبيعية المصنّفة خطأً كهجوم",
         f"{r['fp_count']} من {r['n_normal']} نافذة طبيعية"),
        (f"{ev['latency_median_s']:.0f} s", "Detection Latency", "الوقت من بداية الهجوم حتى التنبيه",
         "الوسيط، بعد 3 نوافذ متتالية"),
    ]
    y = 0.715
    for big, name_en, expl, src in kpis:
        fig.text(0.965, y, big, ha="right", va="center", color=c["sig"], fontsize=48, fontweight="bold")
        fig.text(0.965, y - 0.070, name_en, ha="right", color=c["ink"], fontsize=19, fontweight="bold")
        fig.text(0.965, y - 0.105, ar(expl), ha="right", color=c["ink"], fontsize=13)
        fig.text(0.965, y - 0.137, ar(src), ha="right", color=c["muted"], fontsize=12)
        y -= 0.222
    fig.add_artist(matplotlib.lines.Line2D([0.635, 0.635], [0.10, 0.80], color=c["grid"], lw=1.5))

    # comparison (left area)
    ax = fig.add_axes([0.06, 0.16, 0.54, 0.58])
    style(ax, c)
    x = np.arange(2)
    w = 0.34
    ai = [dr * 100, r["fpr"] * 100]
    en = [r["energy_naive"]["dr"] * 100, r["energy_naive"]["fpr"] * 100]
    b1 = ax.bar(x - w / 2, ai, w, color=c["sig"], label=ar("سرمد (ذكاء اصطناعي)"))
    b2 = ax.bar(x + w / 2, en, w, color=c["grey"], label=ar("الكشف بقوة الإشارة فقط (تقليدي)"))
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%" if v >= 1 else f"{v:.2f}%",
                    ha="center", color=c["ink"], fontsize=17, fontweight="bold")
    ax.set_xticks(x, ["Detection Rate\n(higher is better)", "False Positive Rate\n(lower is better)"], fontsize=15, color=c["ink"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, fontsize=14, labelcolor=c["ink"], loc="upper right")
    fig.text(0.33, 0.80, ar("لماذا الذكاء الاصطناعي؟ الحمل الطبيعي الثقيل أقوى من الهجوم،"), ha="center", color=c["muted"], fontsize=14)
    fig.text(0.33, 0.765, ar("فتفشل العتبة التقليدية بينما يميّز سرمد الإيقاع المصطنع"), ha="center", color=c["muted"], fontsize=14)

    fig.text(0.5, 0.035, "(SNR 10 dB)   " + ar(f"{r['n_attack']} نافذة هجوم و{r['n_normal']} نافذة طبيعية، بيانات محاكاة مبنية على خصائص هجمات منشورة"),
             ha="center", color=c["muted"], fontsize=12)
    fig.savefig(OUT / name, dpi=150, facecolor=c["bg"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path("eval/metrics.json").read_text())
    for th, c in THEMES.items():
        fig_summary(m, c, f"0-summary-{th}.png")
        fig_compare(m, c, f"1-ai-vs-energy-{th}.png")
        fig_scenarios(m, c, f"2-per-scenario-{th}.png")
        fig_snr(m, c, f"3-noise-{th}.png")
        if Path("eval/live_run.json").exists():
            fig_timeline(c, f"4-live-timeline-{th}.png")
    print("saved:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")


if __name__ == "__main__":
    main()
