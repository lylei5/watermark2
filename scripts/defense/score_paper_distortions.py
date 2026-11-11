"""Score the complete development-only published-distortion extension."""
import argparse
import json
from pathlib import Path

from absolute_calibration import calibrate_absolute
from score_content_development import score_query, summarize, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.root / "query_manifest_paper20261110.json"
    records = json.loads(manifest.read_text())["rows"]
    if len(records) != 80 or any(r["split"] != "development" for r in records):
        raise ValueError("Expected the complete fixed development extension")
    for row in records:
        if sha(args.root / "queries" / (row["id"] + ".png")) != row["image_sha256"]:
            raise ValueError("Query image changed")
    rows = [score_query(args.root, r) for r in records]
    previous_path = args.prior_report / "genuine_full_domain.json"
    attack_path = args.prior_report / "reprompt_full_domain.json"
    previous = json.loads(previous_path.read_text())["rows"]
    candidates = json.loads(attack_path.read_text())
    implementation = Path(__file__).with_name("content_authentication.py")
    if candidates["score_implementation_sha256"] != sha(implementation):
        raise ValueError("Existing candidate scores use a different model")
    genuine = [r for r in previous + rows if r["method"] == "sembind_scattered"]
    attack = [r for r in candidates["rows"] if r["method"] == "sembind_scattered"]
    if len(genuine) != 160 or len({r["id"] for r in genuine}) != 160 or len(attack) != 16:
        raise ValueError("Incorrect comparison denominators")
    comparisons = []
    # Every threshold uses genuine scores only; candidates are unchanged prior
    # development observations. This is not independent confirmation.
    for target in [.8, .85, .9, .95, None]:
        target_rows = []
        for field in ["ba", "raw_agreement", "content_agreement",
                      "profiled_content_agreement", "profile_content_lower_bound"]:
            if target is None:
                eligible = [r[field] for r in genuine if r["accepted"]]
                calibration = dict(n=len(genuine), base_count=len(eligible),
                    required_count=len(eligible), target_absolute_tpr=len(eligible)/len(genuine),
                    target_label="maximum_base_absolute_tpr", operator=">=", base_required=True,
                    base_tpr=len(eligible)/len(genuine), feasible=bool(eligible),
                    threshold=min(eligible) if eligible else None,
                    calibration_count=len(eligible), calibration_tpr=len(eligible)/len(genuine),
                    tie_overshoot=0)
            else:
                calibration = calibrate_absolute([r[field] for r in genuine],
                                                [r["accepted"] for r in genuine], target)
            row = dict(score=field, **calibration)
            if row["feasible"]:
                threshold = row["threshold"]
                row.update(candidate_n=len(attack), candidate_accepted=sum(
                    r["accepted"] and r[field] >= threshold for r in attack))
            target_rows.append(row)
        ba = target_rows[0]
        for row in target_rows:
            row["matched_count_to_ba"] = bool(row["feasible"] and ba["feasible"]
                and row["calibration_count"] == ba["calibration_count"])
        comparisons.extend(target_rows)
    result = dict(
        scope="exploratory development extension after earlier results; not confirmation",
        new_query_count=len(rows), new_reference_count=len({(r["task"], r["method"]) for r in rows}),
        main_genuine_count=len(genuine), main_reference_count=16, reused_candidate_count=len(attack),
        new_attacks_run=False, confirmation_read=False,
        baseline="SD2.1 DDIM50 project adapter; unchanged earlier single-candidate CVPR2025 Reprompt outputs",
        weighting="Ten genuine conditions per source, equal condition weights; four new conditions plus six previous",
        manifest_sha256=sha(manifest), previous_genuine_sha256=sha(previous_path),
        candidate_score_sha256=sha(attack_path), script_sha256=sha(__file__),
        score_implementation_sha256=sha(implementation), summary=summarize(rows),
        combined_sembind_summary=summarize(genuine), comparisons=comparisons, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(summary=result["summary"], comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
