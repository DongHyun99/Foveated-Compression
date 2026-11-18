#!/usr/bin/env python3
"""The official per-benchmark metric, in one place (Sec. 4.1, App. A.2).

Every accuracy number this repository reports goes through `official_score`.
Nothing scores with loose substring matching, and the reason is measured: on
the multiple-choice benchmarks the gold answer is a single option letter, and
the letter "A" appears in roughly 57% of all generations, so a substring test
credits more than half of the WRONG answers. An earlier generation of this
work reported exactly that inflated number.

The metrics implemented here, each matching its benchmark's own definition:

  gqa            exact match, ignoring case and punctuation
  pope           exact match on yes/no
  chartqa        relaxed correctness -- 5% numeric tolerance, else exact
  docvqa         ANLS, maximised over the FULL reference-answer list
  textvqa        the VQA leave-one-out consensus formula, averaged over
                 annotators, with the standard answer normalisation
  ocrbench       dataset-conditional containment against the answer list,
                 summed to a raw score out of 1000 (the benchmark's own
                 convention, NOT a percentage)
  mme            the official paired yes/no parse; aggregated as
                 (accuracy + accuracy_plus) * 100 per category, summed to a
                 raw score out of 2800, NOT a percentage
  mmstar         option-letter extraction; aggregated by averaging WITHIN
                 each of the 18 l2 categories first and then across them
  mmbench_en/cn  option-letter extraction, API-free
  scienceqa_img  option-letter extraction

Benchmark-level aggregation (MME's pairing, MMStar's l2 categories, the macro
average with MME divided by 28) lives in `evaluation.evaluate.aggregate`.

TWO THINGS TO STATE WHEN QUOTING A NUMBER FROM HERE.

  1. MME and OCRBench are absolute scores (/2800, /1000), not percentages.
  2. MMBench is scored WITHOUT the official GPT-4 judge and without the
     circular multi-permutation protocol, so it is not directly comparable to
     a GPT-judged leaderboard entry. The extractor used instead is the same
     static regex idea the benchmark's own API-free path uses.

There is one other deliberate deviation. ScienceQA's own parser accepts only a
lowercase-exact match or the pattern "^[A-Z]\\.", so it marks "(B) Haiti"
wrong; this repository renders options as "(A) text" and the model answers in
that format, so scoring with a parser tuned to a different prompt would
measure prompt formatting rather than correctness.
"""
import re
from collections import defaultdict

KNOWN_TASKS = sorted([
    "gqa", "chartqa", "docvqa_val", "pope", "textvqa", "ocrbench",
    "mme", "mmstar", "mmbench_en", "mmbench_cn", "scienceqa_img",
    "train_gqa", "train_textvqa", "train_docvqa", "train_infovqa",
    "train_chartqa", "train_scienceqa",
], key=len, reverse=True)

MC_TASKS = ("mmstar", "mmbench_en", "mmbench_cn", "scienceqa_img")
TRAIN_SPLIT_TASKS = tuple(t for t in KNOWN_TASKS if t.startswith("train_"))


def task_of(name):
    """Map a record name to its task.

    Names are "{task}_{row}" (GQA additionally carries its image id). The
    longest-first ordering of KNOWN_TASKS is load-bearing: "docvqa" is a
    prefix of "docvqa_val", so a shortest-first scan would mis-assign every
    DocVQA row.
    """
    return next((t for t in KNOWN_TASKS if name.startswith(t + "_")), "unknown")


def row_index(name, task):
    """Row index back into the task's own dataset split."""
    return int(name[len(task) + 1:])


# --------------------------------------------------------------------------
# normalisation shared by the VQA-style metrics
# --------------------------------------------------------------------------
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Im": "I'm", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's",
    "lets": "let's", "maam": "ma'am", "mightve": "might've", "mustve": "must've",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's",
    "theyd": "they'd", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youd": "you'd", "youre": "you're", "youve": "you've",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = ";/[]\"{}()=+\\_-><@`,?!."
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_vqa_answer(text):
    """The standard VQA answer normalisation.

    Lowercase, strip articles, expand contractions, map spelled-out digits to
    digits, and remove punctuation except inside numbers. Without it, "2" and
    "two" and "Two." are three different answers and every VQA-style metric
    under-counts.
    """
    t = str(text).replace("\n", " ").replace("\t", " ").strip().lower()
    t = _COMMA_STRIP.sub(r"\1\3", t)
    for p in _PUNCT:
        t = t.replace(p, "" if p != "-" else " ")
    t = _PERIOD_STRIP.sub("", t)
    words = []
    for w in t.split():
        w = _NUMBERS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def exact_match(pred, answer, ignore_case=True, ignore_punctuation=True):
    def norm(s):
        s = str(s).strip()
        if ignore_case:
            s = s.lower()
        if ignore_punctuation:
            s = s.translate(str.maketrans("", "", _PUNCT))
        return " ".join(s.split())

    return 1.0 if norm(pred) == norm(answer) else 0.0


# --------------------------------------------------------------------------
# per-benchmark metrics
# --------------------------------------------------------------------------
def anls(pred, answers, threshold=0.5):
    """Average Normalised Levenshtein Similarity, maximised over `answers`.

    Below the threshold the score is ZERO, not the raw similarity: the metric
    is deliberately discontinuous so that a nearly-right OCR transcription
    scores well while a vaguely similar wrong one scores nothing.

    Maximising over the FULL reference list matters. DocVQA ships several
    equally valid transcriptions per question, and collapsing to answers[0]
    under-counts by several points.
    """
    if isinstance(answers, str):
        answers = [answers]
    p = " ".join(str(pred).lower().strip().split())
    best = 0.0
    for a in answers:
        g = " ".join(str(a).lower().strip().split())
        d = _levenshtein(p, g)
        m = max(len(p), len(g))
        sim = 1.0 - d / m if m else 1.0
        best = max(best, sim)
    return best if best >= threshold else 0.0


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def relaxed_correctness(pred, target, max_relative_change=0.05):
    """ChartQA's metric: 5% relative tolerance on numbers, exact otherwise.

    A chart is read by eye, so demanding an exact float is a measure of
    rendering precision rather than comprehension. Non-numeric targets fall
    back to a normalised exact match, because there is no meaningful notion of
    "within 5%" of a category label.
    """
    def to_float(s):
        s = str(s).strip()
        try:
            if s.endswith("%"):
                return float(s.rstrip("%")) / 100.0
            return float(s.replace(",", ""))
        except ValueError:
            return None

    p, t = to_float(pred), to_float(target)
    if p is not None and t is not None:
        if t == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(p - t) / abs(t) <= max_relative_change else 0.0
    return 1.0 if str(pred).strip().lower() == str(target).strip().lower() else 0.0


def vqa_consensus_score(pred, answers):
    """The VQA leave-one-out consensus formula.

    For each annotator, score min(matches among the OTHER annotators / 3, 1),
    then average. Leave-one-out is what stops a single idiosyncratic
    annotation from being self-confirming.
    """
    res = normalize_vqa_answer(pred)
    norm = [normalize_vqa_answer(a) for a in answers]
    if not norm:
        return 0.0
    acc = []
    for i in range(len(norm)):
        other = [norm[j] for j in range(len(norm)) if j != i]
        acc.append(min(1.0, sum(1 for a in other if a == res) / 3.0))
    return sum(acc) / len(acc)


def ocrbench_score(pred, gt_ans, dataset_name):
    """Containment against the answer list, whitespace-sensitive per dataset.

    HME100k is handwritten mathematical expressions, where spaces carry no
    meaning and are stripped entirely; everything else keeps word boundaries
    and only case-folds. OCRBench's headline figure is the SUM of these over
    1,000 items -- a raw score out of 1000, not a percentage.
    """
    answers = gt_ans if isinstance(gt_ans, (list, tuple)) else [gt_ans]
    if dataset_name == "HME100k":
        def norm(s):
            return str(s).strip().replace("\n", " ").replace(" ", "")
    else:
        def norm(s):
            return str(s).lower().strip().replace("\n", " ")
    p = norm(pred)
    return 1.0 if any(norm(a) in p for a in answers) else 0.0


def parse_mme_answer(pred):
    """MME's official yes/no parse.

    MME pairs every image with one question whose answer is yes and one whose
    answer is no, and its headline metric rewards getting BOTH right. A model
    that always answers "yes" therefore scores at chance on accuracy and zero
    on accuracy_plus, which is the whole point of the pairing.
    """
    p = str(pred).lower().strip().replace(".", "").replace(",", "")
    words = p.split()
    if "yes" in words or p.startswith("yes"):
        return "yes"
    if "no" in words or p.startswith("no"):
        return "no"
    return "other"


def mme_aggregate(rows):
    """(accuracy + accuracy_plus) * 100 per category, summed. Raw score /2800.

    `rows` is an iterable of (category, question_id, is_correct). accuracy is
    per question; accuracy_plus requires BOTH questions of a pair correct.
    """
    per_cat = defaultdict(list)
    for category, pair_id, correct in rows:
        per_cat[category].append((pair_id, bool(correct)))
    total = 0.0
    for category, items in per_cat.items():
        by_pair = defaultdict(list)
        for pair_id, correct in items:
            by_pair[pair_id].append(correct)
        n = sum(len(v) for v in by_pair.values())
        if not n:
            continue
        acc = sum(sum(v) for v in by_pair.values()) / n
        acc_plus = sum(1 for v in by_pair.values() if all(v)) / len(by_pair)
        total += (acc + acc_plus) * 100
    return total


_MC_PATTERNS = (
    re.compile(r"^\s*\(?([A-H])\)?\s*[.:)]?\s*$"),
    re.compile(r"^\s*\(?([A-H])\)?[.:)]\s"),
    re.compile(r"answer\s*(?:is)?\s*:?\s*\(?([A-H])\)?", re.IGNORECASE),
    re.compile(r"option\s*\(?([A-H])\)?", re.IGNORECASE),
)


def extract_mcq_answer(pred, choices=None):
    """Pull an option letter out of a free-form generation.

    Ordered from most to least constrained, so a bare "B" and a sentence
    ending in "the answer is B" both resolve while an incidental capital
    letter in prose does not. Returns None when nothing matches, which scores
    as wrong -- deliberately, since guessing here is what inflates MCQ
    accuracy.
    """
    text = str(pred).strip()
    allowed = set(choices) if choices else {chr(c) for c in range(ord("A"), ord("H") + 1)}
    for pat in _MC_PATTERNS:
        m = pat.search(text)
        if m and m.group(1).upper() in allowed:
            return m.group(1).upper()
    # last resort: a standalone capital letter token
    for tok in re.findall(r"\b([A-H])\b", text):
        if tok.upper() in allowed:
            return tok.upper()
    return None


def mmstar_aggregate(rows):
    """Average WITHIN each l2 category, then across categories.

    MMStar's 18 val categories hold between 31 and 146 samples each, so a flat
    per-sample mean over-weights the largest ones relative to the official
    metric. `rows` is an iterable of (l2_category, score).
    """
    by_cat = defaultdict(list)
    for cat, score in rows:
        by_cat[cat].append(score)
    means = [sum(v) / len(v) for v in by_cat.values()]
    return sum(means) / len(means) if means else 0.0


# --------------------------------------------------------------------------
# the single entry point
# --------------------------------------------------------------------------
def official_score(name, task, pred, answer, extra=None):
    """Per-item score under `task`'s real metric.

    `extra` supplies the fields a single answer string cannot carry -- the
    full annotator answer list for DocVQA and TextVQA, the source dataset name
    for OCRBench. Passing None for a task that needs one raises rather than
    silently falling back to a weaker metric.
    """
    if task in TRAIN_SPLIT_TASKS:
        return _train_split_score(task, pred, answer)
    if task in MC_TASKS:
        choices = ["A", "B", "C", "D"] if task == "mmstar" else None
        return 1.0 if extract_mcq_answer(pred, choices) == str(answer).strip().upper() else 0.0
    if task == "mme":
        gt = str(answer).lower().strip().replace(".", "")
        return 1.0 if parse_mme_answer(pred) == gt else 0.0
    if task == "gqa":
        return exact_match(pred, answer)
    if task == "pope":
        return 1.0 if str(pred).lower().strip() == str(answer).lower().strip() else 0.0
    if task == "chartqa":
        return 1.0 if relaxed_correctness(pred, answer) else 0.0
    if task == "docvqa_val":
        return anls(pred, _need(extra, "answers", task))
    if task == "textvqa":
        return vqa_consensus_score(pred, _need(extra, "answers", task))
    if task == "ocrbench":
        return ocrbench_score(pred, _need(extra, "answers", task),
                              _need(extra, "dataset", task))
    return exact_match(pred, answer)


def _need(extra, key, task):
    if not extra or key not in extra:
        raise ValueError(f"{task} needs extra[{key!r}]; scoring it without that field "
                         f"would silently substitute a weaker metric")
    return extra[key]


def _train_split_score(task, pred, answer):
    """Score a TRAIN-split item from the answer the sampler carried with it.

    The eval path re-fetches multi-answer fields by row index, which only ever
    indexes the EVAL splits -- a train item would index into the wrong
    dataset. Train items therefore carry their own answer and are scored with
    the same metric their matching eval task uses.

    The merger's objective never reads a label (KL against the teacher's own
    answer distribution); the selector's K=1 intervention labels do, and they
    are scored here with the metric of the matching eval task.
    """
    if task == "train_scienceqa":
        return 1.0 if extract_mcq_answer(pred) == str(answer).strip().upper() else 0.0
    if task == "train_gqa":
        return exact_match(pred, answer)
    answers = list(answer) if isinstance(answer, (list, tuple)) else [answer]
    if task == "train_textvqa":
        return vqa_consensus_score(pred, answers)
    if task in ("train_docvqa", "train_infovqa"):
        return anls(pred, answers)
    if task == "train_chartqa":
        return 1.0 if relaxed_correctness(pred, str(answers[0])) else 0.0
    return 1.0 if str(pred).lower().strip() == str(answers[0]).lower().strip() else 0.0
