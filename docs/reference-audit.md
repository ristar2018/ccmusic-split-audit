# Reference audit

Date: 2026-11-12. Scope: the 29 entries of `paper/refs.bib` as printed in
`main_sn.bbl`.

## How each entry was checked

* **20 entries with a DOI**: `https://api.crossref.org/works/{doi}`, comparing title,
  container title, volume, issue, pages, year and the author list field by field.
* **9 entries without a registered DOI**: the publisher page --- PMLR volumes 37, 70
  and 97; JMLR volume 17; the ISMIR 2017 proceedings (two papers); SMC 2017; SMC
  2020; and OpenReview for ICLR 2024.
* dblp, Semantic Scholar and the Zenodo API refused automated access during this
  audit (HTTP 429 and a robots exclusion), so they were not used as sources.

Every entry exists, and every author, year, volume and page number agrees with the
publisher record after the corrections below. No entry is fabricated and no DOI
resolves to a different work. Two Crossref comparisons that looked like mismatches
were artefacts of the comparison: the container title of `downie2010mirex` is the
book series (`Studies in Computational Intelligence`) rather than the book, and
`gebru2018` prints `86--92`, which the check read as `86-92`.

## Per-entry result

| entry | type | as printed | checked against | result |
| --- | --- | --- | --- | --- |
| `brookshire2024eeg` | article | Frontiers in Neuroscience 18:1373515 | Crossref | ok |
| `choi2017` | inproceedings | ISMIR 2017, pp 141-149 | ISMIR proceedings | ok |
| `defferrard2016` | inproceedings | ISMIR 2017, pp 316-323 | Zenodo paper PDF, first page | pages added |
| `downie2010mirex` | incollection | Advances in Music Information Retrieval, pp 93-115 | Crossref | ok |
| `elizalde2024clap` | inproceedings | ICASSP 2024, pp 336-340 | Crossref | ok |
| `engel2017` | inproceedings | ICML 2017, PMLR 70:1068-1077 | PMLR volume 70 | author order fixed |
| `ganin2014` | inproceedings | ICML 2015, PMLR 37:1180-1189 | PMLR volume 37 | url removed |
| `ganin2015` | article | JMLR 17(59):1-35 | JMLR volume 17 | issue number added |
| `gebru2018` | article | Communications of the ACM 64(12):86-92 | Crossref | ok |
| `gemmeke2017audioset` | inproceedings | ICASSP 2017, pp 776-780 | Crossref | ok |
| `hershey2017cnn` | inproceedings | ICASSP 2017, pp 131-135 | Crossref | ok |
| `ji2023leakage` | article | ACM TOIS 41(3):1-27 | Crossref | ok |
| `kapoor2023leakage` | article | Patterns 4(9):100804 | Crossref | ok |
| `kaufman2012` | article | ACM TKDD 6(4):1-21 | Crossref | ok |
| `kong2020panns` | article | IEEE/ACM TASLP 28:2880-2894 | Crossref | ok |
| `kong2022giantmidi` | article | TISMIR 5(1):87-98 | Crossref | ok |
| `alexnet2012` | article | Communications of the ACM 60(6):84-90 | Crossref | ok |
| `lee2017` | inproceedings | SMC 2017, pp 220-226 | self-citation in the authors' ICML 2017 paper | pages added |
| `li2023` | inproceedings | ICLR 2024 | OpenReview w3YZ9MSlBu | ok |
| `swin2021` | inproceedings | ICCV 2021, pp 9992-10002 | Crossref | ok |
| `piczak2015` | inproceedings | ACM MM 2015, pp 1015-1018 | Crossref | ok |
| `recht2019` | inproceedings | ICML 2019, PMLR 97:5389-5400 | PMLR volume 97 | url removed |
| `saito2015pr` | article | PLOS ONE 10(3):e0118432 | Crossref | ok |
| `salamon2017augment` | article | IEEE Signal Processing Letters 24(3):279-283 | Crossref | ok |
| `salamon2014urbansound` | inproceedings | ACM MM 2014, pp 1041-1044 | Crossref | title case |
| `sturm2012gtzan` | inproceedings | ACM MM 2012 workshop, pp 7-12 | Crossref | title case |
| `sturm2014horse` | article | IEEE TMM 16(6):1636-1644 | Crossref | ok |
| `won2020` | inproceedings | SMC 2020 | arXiv 2006.00751 (comment field) | unverifiable pages removed |
| `zhou2025ccmusic` | article | TISMIR 8(1):22-38 | Crossref | ok |

## What was corrected

1. `engel2017`: the author list had Eck and Simonyan ahead of Norouzi. PMLR volume 70
   lists Engel, Resnick, Roberts, Dieleman, Norouzi, Eck, Simonyan. The printed list
   truncates after three authors, so the error was invisible in the PDF.
2. `defferrard2016` and `lee2017` had no page numbers; 316-323 and 220-226 were added.
3. `won2020`: the page range 331-337 could not be confirmed against any source, so it
   was removed rather than left standing.
4. `ganin2015` is JMLR 17(59):1-35; the issue number was missing.
5. Sixteen `pages` fields carried a Unicode en dash instead of `--`. The style file
   tests for an ASCII hyphen when deciding between `p` and `pp`, so seven
   proceedings entries were printing `p 336-340` for a range.
6. Three entries carried a `url` field and printed `URL https://proceedings.mlr.press`
   while the other 26 did not; the fields were removed for consistency.
7. Three ACM proceedings titles were in Crossref's lower-case form and are now title
   case.
8. The header comment claimed every entry body came from Crossref or the arXiv API;
   it now says what actually happened.

## Two references added

The manuscript uses a recording-cluster bootstrap in 20 places and Holm's correction
in three, and cited neither. Both are now cited at their first occurrence in Section 5
(Statistics): Field and Welsh (2007) for bootstrapping clustered data and Holm (1979)
for the sequentially rejective procedure. Adding them pushed the last entry onto a
page of its own, so `\bibsep` is set to 0.75em instead of the class default of 1em;
the list stays on 31 pages.

## What the style file does, and what it cannot do

* Numbering is alphabetical, not order of first citation. `sn-basic.bst` sorts in
  `presort`, so a first citation numbered [13] is the template behaving as designed.
* `@incollection` page ranges always print `p`, never `pp`: `bbl.incoll.pages` is
  hard-coded to `p` at line 273 of the style file and `format.incoll.pages` uses it
  for multi-page ranges too. `downie2010mirex` therefore prints `p 93-115`. This
  cannot be changed from the `.bib` and was left alone.
* More than three authors are truncated to three plus `et al`; journal names are not
  abbreviated.

## Citation hygiene

31 entries, 34 `\cite` commands, every entry cited at least once, no undefined
reference, and `bibtex` reports no warning. Twenty-five entries are cited once; the
most cited is the audited framework itself, three times.

