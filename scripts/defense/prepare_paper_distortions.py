"""Prepare development views at SemBind Tables 5/6 stated distortion settings.

No score or attack candidate is read. PIL is the declared implementation;
this is not a claim that unreleased distortion code has been reproduced.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageFilter, __version__ as pillow_version


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    yield "jpeg70", Image.open(buffer).convert("RGB")
    yield "gaussian_blur3", image.filter(ImageFilter.GaussianBlur(3))
    yield "median7", image.filter(ImageFilter.MedianFilter(7))
    yield "resize50", image.resize((256, 256), Image.Resampling.BICUBIC).resize((512, 512), Image.Resampling.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest = root / "query_manifest_paper20261110.json"
    if manifest.exists():
        raise FileExistsError("The development manifest is already frozen")
    plan = json.loads((root / "generation_plan.json").read_text())
    tasks = [row for row in plan["rows"] if row["split"] == "development"]
    if len(tasks) != 16:
        raise ValueError("Expected all 16 fixed development prompts")
    rows = []
    for row in tasks:
        task = row["task"]
        methods = ["sembind_scattered"] + (["gs_control"] if row["split_index"] < 4 else [])
        for method in methods:
            ref = root / "references" / f"{task:03}_{method}.png"
            image = Image.open(ref).convert("RGB")
            if image.size != (512, 512):
                raise ValueError("Unexpected reference geometry")
            for label, query in variants(image):
                ident = f"{task:03}_{method}_paper20261110_{label}"
                dest = root / "queries" / (ident + ".png")
                if dest.exists():
                    raise FileExistsError(dest)
                query.save(dest)
                rows.append(dict(task=task, split="development", method=method,
                    label=label, id=ident, reference_sha256=sha(ref), image_sha256=sha(dest)))
    protocol = dict(
        scope="post-initial-pilot development extension; no confirmation claims",
        source="https://proceedings.mlr.press/v306/zhang26hj.html",
        source_location="Tables 5 and 6: JPEG Q70, GauBlur radius3, Median kernel7, Resize x0.5",
        implementation="PIL JPEG default subsampling; GaussianBlur(3); MedianFilter(7); bicubic 512->256->512",
        omitted="Brightness factor unspecified in table; existing clean/noise0.01 views remain separately available",
        grouping="all 16 development SemBind references and all 4 GS controls; no score-based selection",
        prior_information="Earlier genuine and Reprompt development results have been seen; this is exploratory",
        comparison="Report all four distortion strata; no candidate-based parameter or subset selection",
        pipeline="existing SD2.1 DDIM50 references; separate from DPM component audit",
        pillow=pillow_version, script_sha256=sha(Path(__file__)),
        plan_sha256=sha(root / "generation_plan.json"), rows=rows)
    manifest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("PREPARED", len(rows))


if __name__ == "__main__":
    main()
