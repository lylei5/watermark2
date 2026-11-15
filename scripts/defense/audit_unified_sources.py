"""Independently check and freeze selected source IDs/prompts; CPU249 only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    run = root/'runs/unified_main_sources_20261116'
    assert not (run/'protocol.json').exists()
    selected = json.loads((run/'selection.json').read_text())
    review = json.loads((run/'prompt_review.json').read_text())
    assert review['selection_sha256'] == sha(run/'selection.json')
    assert review['reviewed_indices'] == list(range(500)) and review['excluded_indices'] == []
    assert selected['skipped_history_files'] == []
    for section in ['source_files', 'tokenizer_files', 'history_manifest_sha256']:
        for path, digest in selected[section].items():
            assert sha(path) == digest, path
    sys.path.insert(0, str(root/'envs/phase1_extra'))
    import pyarrow.parquet as pq
    from transformers import CLIPTokenizer
    token_path = Path(next(iter(selected['tokenizer_files']))).parent
    tokenizer = CLIPTokenizer.from_pretrained(str(token_path), local_files_only=True)
    paths = list(selected['source_files'])
    captions = json.loads(Path(next(p for p in paths if p.endswith('captions_val2017.json'))).read_text())
    first = {}
    for row in sorted(captions['annotations'], key=lambda x: x['id']):
        first.setdefault(row['image_id'], row)
    sdp = pq.read_table(next(p for p in paths if p.endswith('.parquet'))).to_pylist()
    prompt_field = next(k for k in sdp[0] if k.lower() == 'prompt')
    nulls = json.loads(Path(next(p for p in paths if p.endswith('/manifest.json'))).read_text())
    null_ids = {int(x['source_id']) for x in nulls}
    texts, ids = set(), set()

    def scan(obj):
        if isinstance(obj, list):
            for item in obj:
                scan(item)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['caption', 'prompt', 'text', 'source_prompt', 'generation_prompt',
                           'target_prompt', 'attack_prompt', 'prompts', 'captions']:
                    if isinstance(value, str):
                        texts.add(norm(value))
                    elif isinstance(value, list):
                        texts.update(norm(v) for v in value if isinstance(v, str))
                if key in ['image_id', 'coco_id']:
                    try:
                        ids.add(int(value))
                    except (ValueError, TypeError):
                        pass
                if key in ['path', 'url', 'filename', 'file_name'] and isinstance(value, str):
                    ids.update(int(v) for v in re.findall(r'(\d{12})\.(?:jpg|png)', value))
                scan(value)
    for path in selected['history_manifest_sha256']:
        scan(json.loads(Path(path).read_text()))

    def effective(text):
        # Independently exercise the tokenizer's actual truncation path.
        return tuple(tokenizer(text, truncation=True, max_length=77)['input_ids'][1:-1])

    def grams(tokens):
        return {tokens[i:i+5] for i in range(len(tokens)-4)} if len(tokens) >= 10 else set()

    old_tokens = [effective(text) for text in sorted(texts)]
    old_effective = set(old_tokens)
    old_grams = [grams(t) for t in old_tokens]
    unique, current_texts, current_tokens, current_grams = set(), set(), set(), []
    brute_force_pairs = 0
    for expected_index, row in enumerate(selected['sources']):
        assert row['index'] == expected_index
        identity = (row['dataset'], row['source_id'])
        assert identity not in unique
        unique.add(identity)
        if row['dataset'] == 'COCO':
            original = first[int(row['source_id'])]
            assert row['image_id'] == original['image_id']
            assert row['caption_id'] == original['id'] and row['prompt'] == original['caption']
            assert row['image_id'] not in ids | null_ids
        else:
            assert row['dataset'] == 'SDP' and row['prompt'] == sdp[int(row['source_id'])][prompt_field]
        normalized = norm(row['prompt'])
        tokens = effective(row['prompt'])
        assert normalized not in texts | current_texts
        assert tokens not in old_effective | current_tokens
        assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == row['effective_clip_tokens_sha256']
        assert not re.search(selected['content_filter'], row['prompt'], re.I)
        current = grams(tokens)
        for previous in old_grams + current_grams:
            brute_force_pairs += 1
            if current and previous:
                assert 5 * len(current & previous) < 4 * len(current | previous)
        current_texts.add(normalized)
        current_tokens.add(tokens)
        current_grams.append(current)
    assert Counter(row['dataset'] for row in selected['sources']) == {'COCO': 250, 'SDP': 250}
    audit = dict(completed_at=datetime.now(timezone.utc).isoformat(), device='CPU249',
                 sources=500, unchanged_source_text_checks=500, effective_token_checks=500,
                 historical_metadata_files=len(selected['history_manifest_sha256']),
                 historical_prompts=len(texts), historical_coco_ids=len(ids),
                 brute_force_prompt_pairs=brute_force_pairs, overlap_violations=0,
                 source_hash_failures=0, null_image_id_overlap=0,
                 selection_sha256=sha(run/'selection.json'), review_sha256=sha(run/'prompt_review.json'),
                 audit_script_sha256=sha(__file__), model_inferences=0, attacks=0)
    write(run/'audit.json', audit)
    protocol = dict(frozen_at=datetime.now(timezone.utc).isoformat(), scope='Source cohort only; detector and attack protocols freeze separately',
                    independent_source_count=500, strata={'COCO': 250, 'SDP': 250},
                    sources=selected['sources'], selection_sha256=sha(run/'selection.json'),
                    audit_sha256=sha(run/'audit.json'), review_sha256=sha(run/'prompt_review.json'),
                    source_policy='All500 enter each applicable arm; no clean-detection or attack-outcome filtering; candidate/transform replicates clustered by source',
                    generation_seed_policy='Public preview/ordinary generation seeds; watermark keys use separate private provider material',
                    selected_source_tokens='CLIP max_length77 including BOS/EOS; preserve original strings',
                    method_tuning_on_these_sources=False, threshold_tuning_on_these_sources=False)
    write(run/'protocol.json', protocol)
    print(json.dumps(dict(audit=audit, protocol_sha256=sha(run/'protocol.json')), indent=2))


if __name__ == '__main__':
    main()
