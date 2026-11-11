"""Bind every quantitative prose claim to the artifact cell it comes from.

The 337-item gate in verify_paper_numbers.py checks that a printed number exists
in the manuscript. It cannot see whether the number is attached to the right
qualifier -- the bel canto baseline was printed as "low 0.9s" while the artifact
cell said 0.83, and no string check could catch that.

This gate closes that gap with an explicit ledger. Each row states
    artifact file, row selector, column  ->  printed value, prose phrase, qualifier
and the script enforces all three legs:
    (1) the artifact cell still evaluates to the printed value,
    (2) the phrase still occurs in the named section,
    (3) the qualifier tokens still occur in the same sentence as the phrase.

A row fails if any leg breaks, so a re-run of the pipeline that moves a cell, a
later edit that swaps the wording, or a qualifier that silently drifts all
surface as failures.

Usage
    python code/verify_claim_provenance.py            # run the gate
    python code/verify_claim_provenance.py --dump 06_results
"""
import csv
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEDGER = os.path.join(ROOT, 'data', 'results', 'claim_ledger.csv')

SECTIONS = {
    '02_related': 'paper/sections/02_related.tex',
    '03_framework': 'paper/sections/03_framework.tex',
    '04_materials': 'paper/sections/04_materials.tex',
    '05_protocol': 'paper/sections/05_protocol.tex',
    '06_results': 'paper/sections/06_results.tex',
    '07_discussion': 'paper/sections/07_discussion.tex',
    '08_conclusion': 'paper/sections/08_conclusion.tex',
}

# Unicode forms the manuscript uses after LaTeX expansion.
DASH_MAP = [('---', '\u2014'), ('--', '\u2013'), ('``', '\u201c'), ("''", '\u201d')]


def clean(tex):
    """Strip markup, keep readable prose with typographic characters expanded."""
    s = re.sub(r'(?m)(?<!\\)%.*$', '', tex)
    s = re.sub(r'\\begin\{(table|figure|tabular)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'\\(cite\w*|ref|label|input|include)\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = s.replace('{', ' ').replace('}', ' ').replace('~', ' ')
    for a, b in DASH_MAP:
        s = s.replace(a, b)
    s = s.replace('\\%', '%').replace('$', '')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def sentences(tex):
    return [x.strip() for x in re.split(r'(?<=[.;:])\s+', clean(tex)) if x.strip()]


def norm(text):
    """Whitespace-insensitive comparison form; keeps punctuation and dashes."""
    return re.sub(r'\s+', ' ', text).strip()


def load_rows(path):
    with io.open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def select(artifact, where):
    """Evaluate a whitelist-style row selector over the artifact CSV."""
    rows = load_rows(os.path.join(ROOT, artifact.replace('/', os.sep)))
    if not where or where.strip() in ('', '-'):
        return rows
    keep = []
    for row in rows:
        env = {}
        for k, v in row.items():
            try:
                env[k] = float(v)
            except (TypeError, ValueError):
                env[k] = v
        try:
            if eval(where, {'__builtins__': {}}, env):  # noqa: S307 - local ledger only
                keep.append(row)
        except Exception as exc:  # pragma: no cover - surfaced as a failure
            raise RuntimeError('selector failed: %s (%s)' % (where, exc))
    return keep


def fmt(value, decimals):
    return ('%.' + str(decimals) + 'f') % value


def cell_value(row, column):
    """A column name, or a simple arithmetic expression over the row's columns."""
    if column in row:
        return float(row[column])
    allowed = set(row) | set('0123456789.+-*/() ')
    expr = column
    bad = [c for c in expr if c not in allowed and not c.isalpha()]
    if bad:
        raise ValueError('column expression contains %r' % ''.join(sorted(set(bad))))
    return float(eval(expr, {'__builtins__': {}}, dict(row)))  # noqa: S307 - local ledger


def reduce(values, agg):
    agg = (agg or 'single').strip().lower()
    if agg in ('', 'single'):
        if len(values) != 1:
            raise ValueError('agg=single but selector matched %d rows' % len(values))
        return values[0]
    if agg == 'mean':
        return sum(values) / len(values)
    if agg == 'sum':
        return sum(values)
    if agg == 'max':
        return max(values)
    if agg == 'min':
        return min(values)
    if agg == 'count':
        return float(len(values))
    raise ValueError('unknown agg %r' % agg)


def check_row(row, texts):
    cid = row['claim_id']
    section = row['section']
    art = row['artifact']
    where = row['where']
    column = row['column']
    decimals = int(row['decimals'])
    printed = float(row['printed'])
    phrase = norm(row['phrase'])
    quals = [q.strip() for q in (row.get('qualifier') or '').split('|') if q.strip()]

    hits = select(art, where)
    if not hits:
        return cid, 'SELECTOR', 'selector matched no rows: %s' % where
    agg = (row.get('agg') or 'single').strip()
    try:
        if agg.lower() == 'count':
            recomputed = float(len(hits))
        else:
            values = [cell_value(h, column) for h in hits]
            recomputed = reduce(values, agg)
    except ValueError as exc:
        return cid, 'CELL', str(exc)

    tol = 0.5 * 10 ** (-decimals) + 1e-9
    if abs(recomputed - printed) > tol:
        return cid, 'VALUE', 'artifact %s but prose prints %s' % (
            fmt(recomputed, decimals), fmt(printed, decimals))

    body = texts.get(section)
    if body is None:
        return cid, 'SECTION', 'no text for %s' % section
    if phrase not in body:
        return cid, 'PHRASE', 'phrase not found: %r' % phrase

    # The printed value has to sit inside the bound phrase, unless the prose
    # spells it out in words (recorded explicitly in the ledger).
    spelled = (row.get('spelled') or '').strip().lower() in ('yes', 'y', 'true', '1')
    if not spelled:
        forms = {fmt(printed, decimals), fmt(abs(printed), decimals)}
        if printed >= 0:
            forms.add('+' + fmt(printed, decimals))
        if not any(f in phrase for f in forms):
            return cid, 'PHRASE', 'printed %s does not occur in phrase %r' % (
                fmt(printed, decimals), phrase)

    # Locate the sentence carrying the phrase and require the qualifier there.
    host = [s for s in re.split(r'(?<=[.;:])\s+', body) if phrase in s]
    if not host:
        return cid, 'PHRASE', 'phrase not in any sentence: %r' % phrase
    joined = ' '.join(host)
    for q in quals:
        if q.lower() not in joined.lower():
            return cid, 'QUALIFIER', 'qualifier %r missing from the sentence of %r' % (q, phrase)
    return cid, 'OK', fmt(recomputed, decimals)


def main(argv):
    if '--dump' in argv:
        name = argv[argv.index('--dump') + 1]
        tex = io.open(os.path.join(ROOT, SECTIONS[name].replace('/', os.sep)),
                      encoding='utf-8').read()
        for s in sentences(tex):
            print(s)
        return 0

    texts = {}
    for name, rel in SECTIONS.items():
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        if os.path.exists(p):
            texts[name] = clean(io.open(p, encoding='utf-8').read())

    rows = load_rows(LEDGER)
    failures = []
    for row in rows:
        cid, status, detail = check_row(row, texts)
        if status != 'OK':
            failures.append((cid, status, detail))

    print('claim ledger: %d rows' % len(rows))
    for cid, status, detail in failures:
        print('  [%s] %s  %s' % (status, cid, detail))
    print('claims verified: %d, failed: %d' % (len(rows) - len(failures), len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
