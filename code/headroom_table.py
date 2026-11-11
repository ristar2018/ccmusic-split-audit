"""Rebuild the headroom-normalised effect table from the arm metrics.

The manuscript reports the effect rescaled by the remaining score headroom
(Table "tab:headroom"). Those ten rows were previously only in the .tex file,
with no shipped artifact behind them, so nothing could check them. This script
derives the table from the same two files that carry the raw numbers:

    baseline  = mean over seeds of macro_f1_slice for arm A0      (e1_metrics.csv)
    headroom  = 100 - 100 * baseline, in points
    delta     = delta_slice_pt for A1-A0                          (e1_delta.csv)
    share     = delta / headroom

Output: data/results/headroom.csv
"""
import collections
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'data', 'results', 'headroom.csv')

ORDER = ['CTIS', 'music_genre', 'pianos', 'CNPM', 'bel_canto']
DISPLAY = {'CTIS': 'CTIS', 'music_genre': 'music genre', 'pianos': 'pianos',
           'CNPM': 'CNPM', 'bel_canto': 'bel canto'}
BB = {'alexnet': 'AlexNet', 'swin_t': 'Swin-T'}


def read(rel):
    p = os.path.join(ROOT, rel.replace('/', os.sep))
    with io.open(p, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def main():
    metrics = read('data/results/e1_metrics.csv')
    deltas = read('data/results/e1_delta.csv')

    base = collections.defaultdict(list)
    for r in metrics:
        if r['arm'] == 'A0':
            base[(r['library'], r['bb'])].append(float(r['macro_f1_slice']))

    rows = []
    for r in deltas:
        key = (r['library'], r['bb'])
        if key not in base:
            continue
        baseline = sum(base[key]) / len(base[key])
        headroom = 100.0 - 100.0 * baseline
        delta = float(r['delta_slice_pt'])
        rows.append({
            'library': r['library'],
            'library_display': DISPLAY.get(r['library'], r['library']),
            'backbone': r['bb'],
            'backbone_display': BB.get(r['bb'], r['bb']),
            'n_seeds': len(base[key]),
            'baseline_slice': '%.4f' % baseline,
            'headroom_pt': '%.2f' % headroom,
            'delta_slice_pt': '%.2f' % delta,
            'share_frac': '%.3f' % (delta / headroom),
            'share_pct': '%.1f' % (100.0 * delta / headroom),
        })

    rows.sort(key=lambda x: (ORDER.index(x['library']) if x['library'] in ORDER else 99,
                             x['backbone']))
    cols = ['library', 'library_display', 'backbone', 'backbone_display', 'n_seeds',
            'baseline_slice', 'headroom_pt', 'delta_slice_pt', 'share_frac', 'share_pct']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote %s (%d rows)' % (os.path.relpath(OUT, ROOT), len(rows)))
    for r in rows:
        print('  %-12s %-7s headroom %6s  delta %6s  share %s' % (
            r['library_display'], r['backbone_display'], r['headroom_pt'],
            r['delta_slice_pt'], r['share_frac']))


if __name__ == '__main__':
    main()
