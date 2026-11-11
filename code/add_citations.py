"""Insert citations at their natural home in the manuscript.

Every anchor is an exact sentence fragment that must occur exactly once; the
script asserts the hit count and aborts without writing anything if an anchor is
missing or ambiguous, so a partial patch can never land. Replacement strings are
checked not to re-introduce the anchor.

    python code/add_citations.py
"""

import io
import os
import sys

PROJ = r"D:\VSCode\科研任务\音乐情感研究"
S = os.path.join(PROJ, "paper", "sections")

EDITS = [
    # 1. leakage has a data-mining formulation
    ("02_related.tex",
     "where segments of one recording are treated as separate samples.",
     "where segments of one recording are treated as separate samples. The phenomenon also has a "
     "data-mining formulation: leakage is present whenever a feature is generated after the "
     "target and carries information about it \\citep{kaufman2012}, which is what a window of a "
     "test recording does once it lands in training."),

    # 2. audio corpora: shared evaluation efforts, and the augmentation caveat
    ("02_related.tex",
     "Instrument and genre corpora follow the same convention.",
     "Instrument, genre, and tagging corpora follow the same convention, whether the labels are "
     "genre, tags, or instrument \\citep{defferrard2016,choi2017,won2020,lee2017}, and shared "
     "evaluation campaigns have made cross-paper comparison routine \\citep{downie2010mirex}; "
     "Chinese music corpora are a recent addition to that landscape "
     "\\citep{zhou2025ccmusic,kong2022giantmidi}. The environmental-sound literature has also made "
     "the point quantitatively, showing that augmentation does not rescue a split that leaks "
     "\\citep{piczak2015,salamon2017augment}."),

    # 3. a short subsection on the representation side, which is why features get released
    ("02_related.tex",
     "\\subsection{Dataset audits}",
     "\\subsection{Released representations and what they cost}\n"
     "\n"
     "Release formats changed as audio models grew. Large annotated corpora of audio events and "
     "music \\citep{gemmeke2017audioset,engel2017} made it attractive to publish embeddings or "
     "spectrograms instead of waveforms \\citep{hershey2017cnn,kong2020panns}, and self-supervised "
     "audio models now ship representations that downstream users fine-tune rather than retrain "
     "\\citep{elizalde2024clap,li2023}. Each of these releases moves the split decision upstream: "
     "whoever pre-computes the features also decides what an item is, and a downstream user who "
     "sees one image per row has no reason to suspect that rows are windows.\n"
     "\n"
     "\\subsection{Dataset audits}"),

    # 4. what an audit of a score can claim, and the documentation remedy
    ("02_related.tex",
     "That distinction matters for how the result should be read",
     "Audits of this kind have a second, sharper form: a system can be shown not to listen at all, "
     "in which case the benchmark measures the shortcut rather than the task "
     "\\citep{sturm2014horse}. Re-measuring a leaderboard number under a cleaner protocol and "
     "finding that it moves is likewise an established result \\citep{recht2019}, and dataset "
     "documentation now asks authors to state the split unit rather than leave it to the reader "
     "\\citep{gebru2018}. That distinction matters for how the result should be read"),

    # 5. backbones
    ("05_protocol.tex",
     "and two backbones (AlexNet and Swin-T) under three seeds.",
     "and two backbones (AlexNet \\citep{alexnet2012} and Swin-T \\citep{swin2021}) under three seeds."),

    # 6. why macro-averaging under imbalance
    ("05_protocol.tex",
     "Macro-F1 averages only over classes with non-zero support or",
     "Macro-averaging is the appropriate summary when class support is uneven, for the same reason "
     "that precision--recall summaries are preferred to accuracy and ROC under imbalance "
     "\\citep{saito2015pr}. Macro-F1 averages only over classes with non-zero support or"),

    # 7. gradient reversal provenance
    ("05_protocol.tex",
     "representation (a gradient-reversal layer with weight ramped to 0.02 over two warm-up epochs,",
     "representation (a gradient-reversal layer \\citep{ganin2014,ganin2015} with weight ramped to "
     "0.02 over two warm-up epochs,"),
]

# the old key the sections still carry
RENAME = ("paper/sections/02_related.tex", "ji2023recsysleakage", "ji2023leakage")


def main() -> int:
    text = {}

    def load(p):
        p = os.path.abspath(p)
        if p not in text:
            text[p] = io.open(p, encoding="utf-8").read()
        return text[p]

    for f, old, new in EDITS:
        p = os.path.join(S, f)
        s = load(p)
        c = s.count(old)
        if c == 0 and new in s:
            print("skip  %s: already applied" % f)
            continue
        if c != 1:
            print("ABORT %s: anchor occurs %d times (expected 1)\n  %r" % (f, c, old[:80]))
            return 1
        if new.count(old) > 1:
            print("ABORT %s: replacement duplicates the anchor (%d times)\n  %r"
                  % (f, new.count(old), old[:80]))
            return 1
        text[p] = s.replace(old, new)

    fp, old, new = RENAME
    p = os.path.join(PROJ, fp)
    s = load(p)
    if old in s:
        if s.count(old) != 1:
            print("ABORT rename: %s occurs %d times" % (old, s.count(old)))
            return 1
        text[p] = s.replace(old, new)
    elif new not in s:
        print("ABORT rename: neither %s nor %s found" % (old, new))
        return 1

    for p, s in text.items():
        io.open(p, "w", encoding="utf-8", newline="\n").write(s)
        print("patched", os.path.relpath(p, PROJ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
