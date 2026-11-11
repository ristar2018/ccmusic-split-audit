"""Self-test for the claim-provenance gate.

A gate that never fails proves nothing. This script builds a throwaway copy of
the gate in a temporary directory, mutates the section source in ways a real
revision could go wrong, and checks that exactly the expected claims fail.

    python code/negtest_claim_provenance.py

Exit code 0 means every case behaved as expected.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARTIFACTS = [
    'data/results/claim_ledger.csv',
    'data/results/e1_delta.csv',
    'data/results/e1_metrics.csv',
    'data/results/headroom.csv',
    'data/results/c1/e1_delta.csv',
    'data/results/human_review/round2/a1_minus_c1.csv',
    'data/results/human_review/corrected_recording_metrics.csv',
    'data/results/human_review/corrected_recording_delta.csv',
]
SECTIONS = ['02_related', '03_framework', '04_materials', '05_protocol',
            '06_results', '07_discussion', '08_conclusion']
BS = chr(92)

# (label, target file, (old, new) or None, claim ids expected to fail)
CASES = [
    ('baseline', None, None, []),
    ('bel canto baseline reverted to the old wording',
     'paper/sections/07_discussion.tex',
     ('reaches 0.83 for AlexNet and 0.77 for Swin-T',
      'reaches the low 0.9s for both backbones'),
     ['C68', 'C69']),
    ('ceiling reverted to "saturates at 1.00"',
     'paper/sections/07_discussion.tex',
     ('reaches 1.00 in five of the six seed-by-backbone cells',
      'saturates at 1.00 for most seeds'),
     ['C70']),
    ('headroom share edited by one point',
     'paper/sections/07_discussion.tex',
     ('22--37' + BS + '%', '22--38' + BS + '%'),
     ['C73', 'C74']),
]


def stage(base):
    os.makedirs(os.path.join(base, 'code'))
    shutil.copy(os.path.join(HERE, 'verify_claim_provenance.py'),
                os.path.join(base, 'code', 'verify_claim_provenance.py'))
    for rel in ARTIFACTS:
        dst = os.path.join(base, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel.replace('/', os.sep)), dst)
    for name in SECTIONS:
        dst = os.path.join(base, 'paper', 'sections', name + '.tex')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, 'paper', 'sections', name + '.tex'), dst)


def run(base):
    r = subprocess.run([sys.executable,
                        os.path.join(base, 'code', 'verify_claim_provenance.py')],
                       capture_output=True, cwd=base)
    out = r.stdout.decode('utf-8', errors='replace')
    ids = [ln.split()[1] for ln in out.splitlines() if ln.strip().startswith('[')]
    return r.returncode, out, ids


def main():
    base = tempfile.mkdtemp(prefix='claim_negtest_')
    try:
        stage(base)
        ok = True
        for label, rel, mutate, expect in CASES:
            for name in SECTIONS:
                shutil.copy(os.path.join(ROOT, 'paper', 'sections', name + '.tex'),
                            os.path.join(base, 'paper', 'sections', name + '.tex'))
            if mutate:
                p = os.path.join(base, rel.replace('/', os.sep))
                s = io.open(p, encoding='utf-8', newline='').read()
                old, new = mutate
                if s.count(old) != 1:
                    print('SKIP  %-46s anchor count=%d' % (label, s.count(old)))
                    ok = False
                    continue
                io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
            code, out, ids = run(base)
            if expect:
                hit = code == 1 and all(cid in ids for cid in expect)
            else:
                hit = code == 0 and 'failed: 0' in out
            print('%-5s %-46s exit=%d failed=%s' % (
                'PASS' if hit else 'FAIL', label, code, ','.join(ids) or '-'))
            ok = ok and hit
        print('negative test:', 'all cases behaved as expected' if ok else 'MISMATCH')
        return 0 if ok else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
