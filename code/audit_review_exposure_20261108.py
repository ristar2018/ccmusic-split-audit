"""Recount actual A1 pool exposure and optimizer steps from frozen split CSVs."""
import csv
from pathlib import Path

P = Path('data/splits/v2')
LIBS = ('CTIS', 'music_genre', 'pianos', 'CNPM', 'bel_canto')
def read(lib, part):
    with (P / f'{lib}_{part}.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

print('library,pool_slices,injected_slices,injected_recordings,exposed_test_recordings,test_recordings,steps_A0,steps_A1')
for lib in LIBS:
    pool, test, a0, a1 = (read(lib, name) for name in ('qpool', 'qtest', 'a0_train', 'a1_train'))
    poolids = {r['sample_id'] for r in pool}
    testids = {r['sample_id'] for r in test}
    assert not poolids & testids
    injected = [r for r in a1 if r['sample_id'] in poolids]
    train_recs = {r['recording_canonical'] for r in injected}
    test_recs = {r['recording_canonical'] for r in test}
    assert len(injected) <= len(pool)
    for arm, train in (('A0', a0), ('A1', a1)):
        for backbone in ('alexnet', 'swin_t'):
            for seed in range(3):
                log = Path('runs') / lib / arm / backbone / f'seed{seed}' / 'train_log.csv'
                with log.open(encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.DictReader(f))
                assert len(rows) == 10 and int(rows[-1]['epoch']) == 10, str(log)
                assert int(rows[-1]['step']) == (len(train) // 64) * 10, str(log)
    values = (len(injected), len(train_recs), len(train_recs & test_recs), len(test_recs), (len(a0)//64)*10, (len(a1)//64)*10)
    fmt = lambda n: f'{n:,}'.replace(',', '{,}')
    label = {'music_genre': 'music genre', 'bel_canto': 'bel canto'}.get(lib, lib)
    expected = f'{label} & {fmt(values[0])} & {values[1]} & {values[2]}/{values[3]} & {fmt(values[4])} & {fmt(values[5])} ' + r'\\'
    manuscript = Path('paper/sections/10_supplement.tex').read_text(encoding='utf-8')
    assert manuscript.count(expected) == 1, f'Paper row mismatch: {expected}'
    print(','.join(map(str, (lib, len(pool), *values))))
