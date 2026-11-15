"""Freeze a common, outcome-blind 500-prompt evaluation cohort on server249."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path('/data0/liyanlei/watermark_research')
DOMAIN = 'unified-main-20261116'
BLOCKED = re.compile(
    r'\b(nsfw|nude|naked|nudity|nudes|undressed|unclothed|no clothes|bathhouse|blood|'
    r'topless|bottomless|porn\w*|sex\w*|erotic\w*|fetish\w*|breasts?|nipples?|'
    r'genitals?|penis|vagina|lingerie|bikini|gore|gory|beheaded|decapitat\w*|'
    r'dismember\w*|rape|hentai|loli\w*|shota\w*)\b', re.I)
TEXT_KEYS = {'caption', 'prompt', 'text', 'source_prompt', 'generation_prompt',
             'target_prompt', 'attack_prompt', 'prompts', 'captions'}
HISTORY_NAMES = {'protocol.json', 'generation_plan.json', 'selection.json',
                 'selection_initial.json', 'manifest.json', 'metadata.json',
                 'prompts.json', 'source_manifest.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def history_paths(root, run):
    paths = set()
    for parent in [root/'runs', root/'private_eval']:
        for path in parent.glob('**/*.json'):
            if (path.name in HISTORY_NAMES and len(path.relative_to(parent).parts) <= 3
                    and run not in path.parents):
                paths.add(path)
    # Dataset metadata selection files, excluding entire raw caption/prompt pools.
    for path in (root/'datasets').glob('**/*manifest*.json'):
        if len(path.relative_to(root/'datasets').parts) <= 3:
            paths.add(path)
    return sorted(paths)


def extract_history(value, texts, coco_ids):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEXT_KEYS:
                if isinstance(item, str):
                    texts.add(normalize(item))
                elif isinstance(item, list):
                    texts.update(normalize(x) for x in item if isinstance(x, str))
            if key in {'image_id', 'coco_id'}:
                try:
                    coco_ids.add(int(item))
                except (TypeError, ValueError):
                    pass
            if key in {'filename', 'file_name', 'url', 'path'} and isinstance(item, str):
                coco_ids.update(int(x) for x in re.findall(r'(\d{12})\.(?:jpg|png)', item))
            extract_history(item, texts, coco_ids)
    elif isinstance(value, list):
        for item in value:
            extract_history(item, texts, coco_ids)


class PromptIndex:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = set()
        self.effective = set()
        self.grams = []
        self.postings = defaultdict(set)

    def encode(self, text):
        # CLIP retains BOS, at most75 text tokens, EOS; fingerprint excludes padding.
        raw = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        tokens = tuple(raw[:75])
        grams = {tokens[i:i+5] for i in range(len(tokens)-4)} if len(tokens) >= 10 else set()
        return tokens, grams, len(raw)

    def match(self, text, tokens, grams):
        if normalize(text) in self.texts:
            return 'normalized_prompt_overlap'
        if tokens in self.effective:
            return 'effective_clip_prompt_overlap'
        candidates = set()
        for gram in grams:
            candidates.update(self.postings.get(gram, ()))
        for index in candidates:
            old = self.grams[index]
            # Integer comparison implements Jaccard >=0.8 without rounding ambiguity.
            if 5 * len(old & grams) >= 4 * len(old | grams):
                return 'effective_token_5gram_jaccard_ge_0.8'
        return None

    def add(self, text, tokens, grams):
        self.texts.add(normalize(text))
        self.effective.add(tokens)
        index = len(self.grams)
        self.grams.append(grams)
        for gram in grams:
            self.postings[gram].add(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root == ROOT and root.is_dir(), 'Run numerical/data research on249 only'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only task'
    run = root/'runs/unified_main_sources_20261116'
    run.mkdir(exist_ok=True)
    assert not (run/'selection.json').exists(), 'Never overwrite a selected cohort'
    sys.path.insert(0, str(root/'envs/phase1_extra'))
    import pyarrow.parquet as pq
    from transformers import CLIPTokenizer
    tokenizer_path = root/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c/tokenizer'
    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    old_texts, old_ids, manifest, skips = set(), set(), {}, []
    for path in history_paths(root, run):
        if path.stat().st_size > 25_000_000:
            skips.append({'path': str(path), 'bytes': path.stat().st_size, 'reason': 'over25MB'})
            continue
        extract_history(json.loads(path.read_text()), old_texts, old_ids)
        manifest[str(path)] = sha(path)
    index = PromptIndex(tokenizer)
    for text in sorted(old_texts):
        tokens, grams, _ = index.encode(text)
        index.add(text, tokens, grams)
    captions_path = root/'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
    sdp_path = root/'runs/phase1_contract_20261113/sdp_eval.parquet'
    null_path = root/'datasets/unified_coco_nulls_20261115/manifest.json'
    assert sha(null_path) == '13c09dd26845ccad25d9e553888b60fb2e3283de6fd89aaee00e9508b96917a6'
    null_ids = {int(row['source_id']) for row in json.loads(null_path.read_text())}
    by_image = {}
    for row in sorted(json.loads(captions_path.read_text())['annotations'], key=lambda x: x['id']):
        by_image.setdefault(row['image_id'], row)
    sdp = pq.read_table(sdp_path).to_pylist()
    prompt_key = next(k for k in sdp[0] if k.lower() == 'prompt')
    pools = {
        'COCO': [dict(source_id=str(k), prompt=v['caption'], image_id=k, caption_id=v['id'])
                 for k, v in by_image.items()],
        'SDP': [dict(source_id=str(i), prompt=row[prompt_key]) for i, row in enumerate(sdp)]}
    sources, selection = [], {}
    for dataset, pool in pools.items():
        ordered = sorted(pool, key=lambda x: hashlib.sha256(
            f"{DOMAIN}/{dataset}/{x['source_id']}".encode()).hexdigest())
        accepted, rejected = [], []
        for row in ordered:
            prompt = row['prompt']
            tokens, grams, full_length = index.encode(prompt)
            reason = ('empty_prompt' if not normalize(prompt) else
                      'content_filter' if BLOCKED.search(prompt) else
                      'natural_null_source_overlap' if row.get('image_id') in null_ids else
                      'recorded_coco_id_overlap' if row.get('image_id') in old_ids else
                      index.match(prompt, tokens, grams))
            if reason:
                rejected.append(dict(source_id=row['source_id'], reason=reason))
                continue
            source = dict(index=len(sources) + len(accepted), dataset=dataset, **row,
                          full_clip_text_tokens=full_length,
                          effective_clip_tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                          generation_seed=int.from_bytes(hashlib.sha256(
                              f"{DOMAIN}/generation/{dataset}/{row['source_id']}".encode()).digest()[:8], 'big') % (2**63))
            accepted.append(source)
            index.add(prompt, tokens, grams)
            if len(accepted) == 250:
                break
        assert len(accepted) == 250, (dataset, len(accepted))
        sources.extend(accepted)
        selection[dataset] = dict(pool_size=len(pool), selected=len(accepted), rejected_before_fill=rejected,
                                  rejection_counts=dict(Counter(x['reason'] for x in rejected)))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), domain=DOMAIN,
                  stage='selected_pending_text_review', sources=sources, selection=selection,
                  source_files={str(p): sha(p) for p in [captions_path, sdp_path, null_path]},
                  tokenizer_files={str(p): sha(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file()},
                  history_manifest_sha256=manifest, skipped_history_files=skips,
                  historical_normalized_prompts=len(old_texts), historical_coco_ids=len(old_ids),
                  content_filter=BLOCKED.pattern, script_sha256=sha(__file__),
                  duplicate_rule='NFKC/casefold/space equality OR effective first75 CLIP tokens equality OR token5gram set Jaccard>=0.8 for >=10 effective tokens',
                  source_order='COCO then SDP; SHA256(domain/dataset/source_id); first annotation ID per COCO image',
                  scope='Recorded metadata within this project only; lexical screening is not semantic equivalence or a claim about pretraining; no outcome-based source exclusion',
                  generation_jobs=0, attack_jobs=0)
    write(run/'selection.json', result)
    write(run/'selection_summary.json', dict(sources=len(sources), datasets={k: v['selected'] for k, v in selection.items()},
        selection_sha256=sha(run/'selection.json'), scanned_history_files=len(manifest), skipped_history_files=skips,
        historical_prompts=len(old_texts), historical_coco_ids=len(old_ids),
        rejected={k: v['rejection_counts'] for k, v in selection.items()},
        prompts_exceeding75_tokens=sum(x['full_clip_text_tokens'] > 75 for x in sources),
        null_source_id_overlap=sum(x.get('image_id') in null_ids for x in sources)))
    (run/'prompt_review.txt').write_text('\n'.join(f"{x['index']:03d} [{x['dataset']}:{x['source_id']}] {x['prompt']}" for x in sources) + '\n', encoding='utf-8')
    print((run/'selection_summary.json').read_text())


if __name__ == '__main__':
    main()
