"""Build a minimal, allowlisted set of public visualization assets.

Run in the Reddit AI Topic Explorer clone. Does not stage, commit or push.
The release is not a certification of anonymization or of redistribution rights.
"""
from __future__ import annotations

import json
import re
import base64
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts'
OUT = ROOT / 'public_assets'
EXPECTED_CHARTS = ('topic_keywords.json', 'topic_map.json', 'topic_hierarchy.json', 'document_map.json')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE = re.compile(r'(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect_strings(obj, path='root') -> None:
    """Reject obvious contact information and unusually long text string values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            inspect_strings(value, f'{path}.{key}')
    elif isinstance(obj, list):
        for value in obj:
            inspect_strings(value, path + '[]')
    elif isinstance(obj, str):
        if path.endswith('.bdata'):
            # Plotly stores numeric arrays in base64-encoded typed array payloads.
            require(re.fullmatch(r'[A-Za-z0-9+/=]+', obj) is not None, f'Unexpected Plotly binary data: {path}')
            base64.b64decode(obj, validate=True)
            return
        require(not EMAIL.search(obj), f'Email-like text in {path}')
        require(not PHONE.search(obj), f'Phone-like text in {path}')
        require(len(obj) <= 350, f'Unusually long string in {path}; manually inspect source before publishing')


def main() -> None:
    require(not OUT.exists(), 'public_assets/ already exists. Move it aside before rebuilding; no files overwritten.')
    index = pd.read_parquet(SOURCE / 'public' / 'document_index.parquet')
    freq = pd.read_parquet(SOURCE / 'public' / 'word_frequencies_public.parquet')
    topic = pd.read_csv(SOURCE / 'public' / 'topic_information_public.csv')
    xy = np.load(SOURCE / 'projections' / 'umap_2d.npy', allow_pickle=False)
    z5 = np.load(SOURCE / 'projections' / 'umap_5d.npy', allow_pickle=False)

    require(set(index.columns) == {'subreddit','topic','doc_id'}, 'Unexpected document-index columns')
    require(set(freq.columns) == {'subreddit','ngram','remove_stopwords','term','count','document_frequency'}, 'Unexpected frequency columns')
    require(set(topic.columns) == {'Topic','Count','Name','Representation'}, 'Unexpected topic columns')
    require(len(index) == len(xy) == len(z5) == 4000, 'Document/projection mismatch')
    require(xy.shape == (4000,2) and z5.shape == (4000,5), 'Unexpected projection shape')
    require(np.isfinite(xy).all() and np.isfinite(z5).all(), 'Nonfinite coordinate')
    require(index.doc_id.is_unique and index.doc_id.astype(str).str.fullmatch(r'DOC_\d{4}').all(), 'Bad synthetic ID')
    require(freq['document_frequency'].ge(10).all() and freq['count'].ge(10).all(), 'Frequency below public minimum')
    require(freq['document_frequency'].le(freq['count']).all(), 'Invalid frequency relationship')
    require(topic.Topic.is_unique, 'Topic IDs not unique')
    require(set(index.topic.astype(int)).issubset(set(topic.Topic.astype(int))), 'Unknown topic ID')
    require(index[['subreddit','topic','doc_id']].notna().all().all(), 'Missing index values')
    require(freq.notna().all().all(), 'Missing frequency value')
    require(topic.notna().all().all(), 'Missing topic value')
    for value in freq['term'].astype(str).unique(): inspect_strings(value, 'frequency term')
    for column in ('Name','Representation'):
        for value in topic[column].astype(str): inspect_strings(value, f'topic {column}')
    for value in index.subreddit.astype(str).unique(): inspect_strings(value, 'subreddit')

    charts = {}
    for filename in EXPECTED_CHARTS:
        obj = json.loads((SOURCE / 'bertopic' / 'final_candidate' / 'charts' / filename).read_text(encoding='utf-8'))
        require(set(obj.keys()).issubset({'data','layout','frames'}), f'Unexpected chart structure {filename}')
        if filename == 'document_map.json':
            for trace in obj.get('data', []):
                for key in ('hovertext','text','customdata','hovertemplate','ids'):
                    trace.pop(key, None)
                trace['hoverinfo'] = 'skip'
                trace['mode'] = 'markers'
        inspect_strings(obj, filename)
        charts[filename] = obj

    # Write only files explicitly listed here. Never copy entire artifacts directory.
    (OUT / 'charts').mkdir(parents=True)
    (OUT / 'projections').mkdir()
    index.to_parquet(OUT / 'document_index.parquet', index=False)
    freq.to_parquet(OUT / 'word_frequencies_public.parquet', index=False)
    topic.to_csv(OUT / 'topic_information_public.csv', index=False)
    shutil.copyfile(SOURCE / 'projections' / 'umap_2d.npy', OUT / 'projections' / 'umap_2d.npy')
    shutil.copyfile(SOURCE / 'projections' / 'umap_5d.npy', OUT / 'projections' / 'umap_5d.npy')
    for filename, obj in charts.items():
        (OUT / 'charts' / filename).write_text(json.dumps(obj, ensure_ascii=False, separators=(',',':')), encoding='utf-8')

    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file())
    require(len(files) == 9, 'Incorrect release file count')
    print('PASS: public release generated with these nine allowlisted files:')
    for filename in files: print('  ' + filename)
    print(f'PASS: {len(index)} document labels, {len(freq)} filtered term records, {len(topic)} topic metadata rows')
    print('NOTE: Basic privacy checks passed; rights and contextual reidentification require human review.')


if __name__ == '__main__':
    main()
