"""Plot completed audits with their different model/precision scopes visible."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--prior-report", type=Path, required=True)
    args = p.parse_args()
    paths = [args.report / "dpm_results.json", args.report / "paper_distortions.json",
             args.prior_report / "genuine_full_domain.json", args.prior_report / "reprompt_full_domain.json"]
    dpm, extension, genuine, candidates = [json.loads(p.read_text()) for p in paths]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    blue, orange = "#0072B2", "#D55E00"
    for method, color, marker in [("sembind", blue, "o"), ("gs", orange, "^")]:
        x, y = [], []
        for task in range(4):
            rows = [r for r in dpm["rows"] if r["task"] == task and r["method"] == method]
            x.append(next(r["channel_sign_error"] for r in rows if r["setting"] == "blind_encoder"))
            y.append(next(r["channel_sign_error"] for r in rows if r["setting"] == "blind_decoder"))
        axes[0].scatter(x, y, c=color, marker=marker, s=46, label=method, edgecolors="white", linewidth=.5)
    axes[0].plot([0, .5], [0, .5], color=".5", ls="--", lw=1)
    axes[0].set(xlim=(0,.5), ylim=(0,.5), xlabel="VAE encoder: latent sign error",
                ylabel="VAE optimization: latent sign error", title="a  DPM / fp32: 4 prompts x 2 carriers")
    axes[0].legend(frameon=False, loc="upper left")
    labels = ["jpeg70", "gaussian_blur3", "median7", "resize50"]
    values = [[r["ba"] for r in extension["rows"] if r["method"] == "sembind_scattered"
               and r["label"] == label] for label in labels]
    axes[1].boxplot(values, widths=.48, showfliers=False, medianprops={"color": "black"},
                    boxprops={"color": blue}, whiskerprops={"color": blue}, capprops={"color": blue})
    for i, v in enumerate(values):
        axes[1].scatter(i+1+np.linspace(-.13,.13,len(v)), v, color=blue, s=14, alpha=.65)
    axes[1].axhline(181/256, ls="--", c=".4", lw=1)
    axes[1].set(xticks=range(1,5), xticklabels=["JPEG70", "Blur3", "Median7", "Resize.5"],
                ylabel="Decoded bit accuracy", ylim=(0,1.03), title="b  DDIM: 16 images per distortion")
    g = [r for r in genuine["rows"]+extension["rows"] if r["method"] == "sembind_scattered"]
    a = [r for r in candidates["rows"] if r["method"] == "sembind_scattered"]
    axes[2].scatter([r["ba"] for r in g], [r["profiled_content_agreement"] for r in g],
                    c=blue, marker="o", s=22, alpha=.55, label="Genuine: 16 sources, 160 views")
    axes[2].scatter([r["ba"] for r in a], [r["profiled_content_agreement"] for r in a],
                    c=orange, marker="x", s=44, label="Reprompt: 16 existing candidates")
    axes[2].axvline(181/256, ls="--", c=".4", lw=1)
    axes[2].set(xlim=(0,1.03), ylim=(0,1.03), xlabel="Decoded bit accuracy",
                ylabel="Estimated content agreement", title="c  DDIM: content and carrier evidence")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8)
    fig.text(.5,.015,"Development diagnostics only. Panel a: true sign errors use private source information. "
             "Panels b/c: prior DDIM adapter; repeated views are correlated.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ["png","svg"]:
        fig.savefig(args.report / ("evidence."+ext), dpi=220, bbox_inches="tight")
    plt.close(fig)
    metadata = dict(inputs={str(p):sha(p) for p in paths}, script_sha256=sha(Path(__file__)),
        matplotlib_version=matplotlib.__version__, no_confirmation_data=True,
        interpretation="No superiority or attack content-success claims; source repetitions are not independent",
        png_sha256=sha(args.report/"evidence.png"), svg_sha256=sha(args.report/"evidence.svg"))
    (args.report/"figure_provenance.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
