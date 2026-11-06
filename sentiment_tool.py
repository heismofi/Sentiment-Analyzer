#!/usr/bin/env python3
"""
Sentiment Analysis Tool with NLTK and VADER
===========================================
Classifies social media comments, tweets, or reviews as positive, negative,
or neutral, then reports statistics and saves charts.

Data sources (as per the project brief)
  * A public Twitter sentiment dataset from Kaggle (auto-detected formats:
    Sentiment140, Twitter Entity Sentiment "twitter_training.csv", Twitter US
    Airline Sentiment, or any CSV with a text column), OR
  * The bundled sample social media dataset (data/sample_social_media.csv).

Examples
  python sentiment_tool.py                          # sample data
  python sentiment_tool.py --file Tweets.csv        # Kaggle CSV
  python sentiment_tool.py --text "I love this!"    # one sentence
  python sentiment_tool.py --interactive            # type sentences live
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections import Counter
from pathlib import Path


def _pause():
    """On Windows, keep the console open when the script is started with
    no arguments (e.g. double-clicked) so the output can be read."""
    if sys.platform == "win32" and len(sys.argv) == 1:
        try:
            input("\nPress Enter to close this window...")
        except Exception:
            pass


def _crash_handler(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    if issubclass(exc_type, ModuleNotFoundError):
        print("\nA required package is missing. Install everything with:\n"
              "    pip install nltk pandas matplotlib")
    _pause()


if __name__ == "__main__":
    sys.excepthook = _crash_handler

import pandas as pd  # noqa: E402

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # works when frozen into an .exe
SAMPLE_PATH = HERE / "data" / "sample_social_media.csv"

# The sample data is embedded here so the app still works if the data/ folder is missing.
SAMPLE_ROWS = [
    (1, '2026-10-26', 'Twitter', 'Great value for money, would definitely buy again.', 'positive'),
    (2, '2026-10-27', 'Instagram', 'The exam timetable has been uploaded to the portal.', 'neutral'),
    (3, '2026-10-28', 'Facebook', "Can't believe how easy the checkout was. Well done, designers!", 'positive'),
    (4, '2026-10-29', 'Twitter', 'The package arrived on Tuesday.', 'neutral'),
    (5, '2026-10-30', 'Instagram', 'Registration for the workshop closes on Friday.', 'neutral'),
    (6, '2026-10-31', 'Facebook', 'Going to the market to buy tomatoes and onions.', 'neutral'),
    (7, '2026-11-01', 'Twitter', 'I feel so stressed and tired, this semester is awful.', 'negative'),
    (8, '2026-10-26', 'Instagram', 'Excellent battery life and a gorgeous screen. Highly recommend this tablet.', 'positive'),
    (9, '2026-10-27', 'Facebook', 'Just got my new laptop and it is AMAZING!! Fast, light, and the battery lasts all day 😍', 'positive'),
    (10, '2026-10-28', 'Twitter', 'Customer service was kind and patient with me. Really appreciate it.', 'positive'),
    (11, '2026-10-29', 'Instagram', 'Very disappointed. The product looks nothing like the photos.', 'negative'),
    (12, '2026-10-30', 'Facebook', "My sister just got her dream job and I couldn't be happier for her ❤️", 'positive'),
    (13, '2026-10-31', 'Twitter', "Honestly one of the best concerts I've ever attended, the energy was unreal", 'positive'),
    (14, '2026-11-01', 'Instagram', 'Terrible experience, the website kept timing out during payment.', 'negative'),
    (15, '2026-10-26', 'Facebook', 'The match starts at 8pm tonight.', 'neutral'),
    (16, '2026-10-27', 'Twitter', 'Dropped my laptop off at the repair shop this morning.', 'neutral'),
    (17, '2026-10-28', 'Instagram', 'My order arrived broken and they refuse to refund me. Absolutely disgusting.', 'negative'),
    (18, '2026-10-29', 'Facebook', 'Yeah great, another delayed delivery. Thanks a lot...', 'negative'),
    (19, '2026-10-30', 'Twitter', 'This update ruined the app. It crashes every time I open it!!!', 'negative'),
    (20, '2026-10-31', 'Instagram', 'Today is Wednesday and the weather is cloudy.', 'neutral'),
    (21, '2026-11-01', 'Facebook', 'That movie was hilarious, I laughed the whole way through', 'positive'),
    (22, '2026-10-26', 'Twitter', 'Awful movie. Boring plot and bad acting, I almost fell asleep.', 'negative'),
    (23, '2026-10-27', 'Instagram', 'The power went off again. Honestly sick of this.', 'negative'),
    (24, '2026-10-28', 'Facebook', 'Just posted a new blog about Python lists.', 'neutral'),
    (25, '2026-10-29', 'Twitter', 'Anyone know which bus goes to the city centre?', 'neutral'),
    (26, '2026-10-30', 'Instagram', 'Charging my phone before I head out.', 'neutral'),
    (27, '2026-10-31', 'Facebook', 'Updated my resume and sent it to two companies.', 'neutral'),
    (28, '2026-11-01', 'Twitter', 'The new phone model has a 6.1 inch display and 128GB storage.', 'neutral'),
    (29, '2026-10-26', 'Instagram', "I'm so excited for the weekend, big plans with friends!", 'positive'),
    (30, '2026-10-27', 'Facebook', 'Shoutout to the support team at @TechHub, they fixed my issue in 5 minutes. Great service! #happy', 'positive'),
    (31, '2026-10-28', 'Twitter', "Rent keeps going up and salaries are not. It's depressing.", 'negative'),
    (32, '2026-10-29', 'Instagram', 'The hostel room was dirty and smelled bad. Do not recommend.', 'negative'),
    (33, '2026-10-30', 'Facebook', 'The report is due next Monday.', 'neutral'),
    (34, '2026-10-31', 'Twitter', "I'm really upset that they cancelled the event at the last minute.", 'negative'),
    (35, '2026-11-01', 'Instagram', 'Finally a bank app that just works. Smooth, quick and secure.', 'positive'),
    (36, '2026-10-26', 'Facebook', 'What a beautiful sunrise this morning. Feeling grateful.', 'positive'),
    (37, '2026-10-27', 'Twitter', "I'm reading a book about data structures.", 'neutral'),
    (38, '2026-10-28', 'Instagram', 'The store opens at 9am and closes at 6pm.', 'neutral'),
    (39, '2026-10-29', 'Facebook', 'Battery drains in three hours. What a joke of a phone.', 'negative'),
    (40, '2026-10-30', 'Twitter', 'Watching the news and having tea.', 'neutral'),
    (41, '2026-10-31', 'Instagram', "Today's lecture was actually fun and super helpful, thanks prof!", 'positive'),
    (42, '2026-11-01', 'Facebook', 'Our class group chat has 45 members now.', 'neutral'),
    (43, '2026-10-26', 'Twitter', 'Food was cold, the waiter was rude, never coming back.', 'negative'),
    (44, '2026-10-27', 'Instagram', 'Total waste of money. Stopped working after two days.', 'negative'),
    (45, '2026-10-28', 'Facebook', 'The delivery arrived early and everything was perfectly packed. Thank you!', 'positive'),
    (46, '2026-10-29', 'Twitter', "Best pizza I've had in years. Totally worth the wait :)", 'positive'),
    (47, '2026-10-30', 'Instagram', 'The new camera on this phone is stunning, photos look incredible', 'positive'),
    (48, '2026-10-31', 'Facebook', 'They charged me twice and support keeps ignoring my emails. Unacceptable!', 'negative'),
    (49, '2026-11-01', 'Twitter', 'The conference will take place next month.', 'neutral'),
    (50, '2026-10-26', 'Instagram', 'Slow, buggy and full of ads. Uninstalling this garbage app.', 'negative'),
    (51, '2026-10-27', 'Facebook', "The network has been terrible all week, I'm so frustrated", 'negative'),
    (52, '2026-10-28', 'Twitter', 'This course is brilliant, I finally understand machine learning.', 'positive'),
    (53, '2026-10-29', 'Instagram', 'Worst flight delay of my life, and no one even apologised.', 'negative'),
    (54, '2026-10-30', 'Facebook', 'The library has moved to the second floor.', 'neutral'),
    (55, '2026-10-31', 'Twitter', 'Our team just won the hackathon!!! So proud of everyone 🎉', 'positive'),
    (56, '2026-11-01', 'Instagram', 'Loving the new update, the app feels so much smoother now', 'positive'),
    (57, '2026-10-26', 'Facebook', 'Meeting is scheduled for 3pm in the main hall.', 'neutral'),
    (58, '2026-10-27', 'Twitter', 'I hate waiting in traffic for hours, this city is exhausting 😡', 'negative'),
    (59, '2026-10-28', 'Instagram', 'Five stars. The hotel staff were friendly and the room was spotless.', 'positive'),
    (60, '2026-10-29', 'Facebook', 'Worst customer service ever. Waited two hours and nobody helped me. #fail', 'negative'),
]


def ensure_sample_file() -> Path:
    """Create data/sample_social_media.csv from the embedded rows if it is missing."""
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(SAMPLE_ROWS, columns=["id", "date", "platform", "text", "label"]
                     ).to_csv(SAMPLE_PATH, index=False, encoding="utf-8")
    return SAMPLE_PATH

class DatasetError(Exception):
    """Raised for problems with the input data (shown nicely by CLI and GUI)."""


LABELS = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}

TEXT_CANDIDATES = ["text", "tweet", "tweet_text", "content", "review",
                   "comment", "sentence", "clean_text", "message"]
LABEL_CANDIDATES = ["true_label", "label", "sentiment", "airline_sentiment",
                    "target", "polarity", "category"]
DATE_CANDIDATES = ["date", "tweet_created", "created_at", "timestamp", "time"]

LABEL_MAP = {
    "positive": "positive", "pos": "positive", "4": "positive", "1": "positive",
    "negative": "negative", "neg": "negative", "0": "negative", "-1": "negative",
    "neutral": "neutral", "2": "neutral", "irrelevant": "neutral",
}

STOPWORDS = set("""
a about after again all also am an and any are as at be because been before
being but by can could did do does doing down for from get got had has have
having he her here him his how i if in into is it its just like me more most
my no nor not now of off on once only or other our out over own same she so
some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who why
will with would you your amp rt via
""".split())


# --------------------------------------------------------------------------- #
# 1. VADER setup
# --------------------------------------------------------------------------- #
def get_analyzer():
    """Return a VADER analyzer, downloading the lexicon on first use."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    bundled = HERE / "nltk_data"           # shipped inside the .exe / project folder
    if bundled.exists() and str(bundled) not in nltk.data.path:
        nltk.data.path.insert(0, str(bundled))
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


# --------------------------------------------------------------------------- #
# 2. Text cleaning and classification
# --------------------------------------------------------------------------- #
def clean_text(text) -> str:
    """Light cleaning. Punctuation, CAPS and emoticons are kept because
    VADER uses them as sentiment signals."""
    text = str(text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&(quot|lt|gt);", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # links
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = text.replace("#", "")                        # keep hashtag word
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_from_score(compound: float, threshold: float = 0.05) -> str:
    """Standard VADER convention: >= +0.05 positive, <= -0.05 negative."""
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"


def normalise_label(value):
    """Map dataset labels (Positive, 4, 0, Irrelevant...) to our 3 classes."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    try:
        s = str(int(float(s)))
    except ValueError:
        pass
    return LABEL_MAP.get(s)


def analyze(df: pd.DataFrame, analyzer, threshold: float = 0.05) -> pd.DataFrame:
    """Add clean_text, neg/neu/pos/compound and sentiment columns."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    scores = pd.DataFrame(
        [analyzer.polarity_scores(t) for t in df["clean_text"]], index=df.index
    )
    df = pd.concat([df, scores], axis=1)
    df["sentiment"] = df["compound"].apply(lambda c: label_from_score(c, threshold))
    return df


# --------------------------------------------------------------------------- #
# 3. Data loading
# --------------------------------------------------------------------------- #
def _find_col(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _extract_from_zip(path: Path, log=print) -> Path:
    """Kaggle downloads are .zip archives. Pull the data file out of one."""
    import tempfile
    import zipfile
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise DatasetError(f"'{path.name}' is not a valid ZIP archive.")
    members = [i for i in zf.infolist()
               if not i.is_dir() and "__MACOSX" not in i.filename
               and i.filename.lower().endswith((".csv", ".tsv", ".txt"))]
    if not members:
        names = ", ".join(i.filename for i in zf.infolist()[:8])
        raise DatasetError("No CSV file found inside the ZIP. It contains: " + names)
    # prefer a file with 'train' in its name, otherwise the biggest one
    members.sort(key=lambda i: ("train" not in i.filename.lower(), -i.file_size))
    chosen = members[0]
    log(f"Using '{chosen.filename}' from the ZIP archive")
    return Path(zf.extract(chosen, tempfile.mkdtemp(prefix="sentiment_")))


def load_dataset(path, text_col=None, label_col=None, date_col=None,
                 max_rows=20000, seed=42, log=print) -> pd.DataFrame:
    """Load a Kaggle CSV (or any CSV) and standardise to: text, true_label, date."""
    path = Path(path)
    if path == SAMPLE_PATH:
        ensure_sample_file()
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    if path.suffix.lower() == ".zip":
        path = _extract_from_zip(path, log)
    if path.suffix.lower() in (".xlsx", ".xls", ".json", ".parquet", ".pkl"):
        raise DatasetError(f"'{path.suffix}' files are not supported. Open the file in Excel "
                           "(or similar) and save it as CSV, or download the CSV version.")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    enc = "utf-8" if path == SAMPLE_PATH else "latin-1"

    try:
        peek = pd.read_csv(path, sep=sep, header=None, nrows=5, encoding=enc)
    except Exception as e:
        raise DatasetError(f"Could not read this file as a CSV: {e}")
    entity_labels = {"positive", "negative", "neutral", "irrelevant"}

    if peek.shape[1] == 6 and set(peek[0].astype(str)) <= {"0", "2", "4"}:
        # Sentiment140 (no header): target, ids, date, flag, user, text
        df = pd.read_csv(path, sep=sep, header=None, encoding=enc,
                         names=["target", "ids", "date", "flag", "user", "text"])
        text_col, label_col, date_col = "text", "target", "date"
        log("Detected format: Sentiment140")
    elif peek.shape[1] == 4 and peek[2].astype(str).str.lower().isin(entity_labels).all():
        # Twitter Entity Sentiment (no header): id, entity, sentiment, text
        df = pd.read_csv(path, sep=sep, header=None, encoding=enc,
                         names=["id", "entity", "sentiment", "text"])
        text_col, label_col = "text", "sentiment"
        log("Detected format: Twitter Entity Sentiment")
    else:
        df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
        text_col = text_col or _find_col(df.columns, TEXT_CANDIDATES)
        label_col = label_col or _find_col(df.columns, LABEL_CANDIDATES)
        date_col = date_col or _find_col(df.columns, DATE_CANDIDATES)
        if text_col is None:
            raise DatasetError("Could not find a text column. Set the text column name. "
                     f"Columns: {list(df.columns)}")

    if text_col not in df.columns:
        raise DatasetError(f"Column '{text_col}' not found. Columns: {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    if label_col and label_col in df.columns and label_col != text_col:
        out["true_label"] = df[label_col].apply(normalise_label)
    else:
        out["true_label"] = None
    if date_col and date_col in df.columns:
        cleaned = df[date_col].astype(str).str.replace(r"\s[A-Z]{3,4}\s", " ", regex=True)
        out["date"] = pd.to_datetime(cleaned, errors="coerce", utc=True).dt.tz_localize(None)
    else:
        out["date"] = pd.NaT

    out = out.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    if out.empty:
        raise DatasetError("The file contains no text to analyse.")
    if len(out) > max_rows:
        log(f"Dataset has {len(out):,} rows; sampling {max_rows:,} (use --max-rows to change).")
        out = out.sample(max_rows, random_state=seed)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Evaluation and statistics
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame):
    """Compare VADER's labels with the dataset's human labels (if present)."""
    d = df[df["true_label"].isin(LABELS)]
    if d.empty:
        return None
    cm = (pd.crosstab(d["true_label"], d["sentiment"])
          .reindex(index=LABELS, columns=LABELS, fill_value=0))
    rows = {}
    for lab in LABELS:
        tp = cm.loc[lab, lab]
        fp = cm[lab].sum() - tp
        fn = cm.loc[lab].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows[lab] = {"precision": p, "recall": r, "f1": f1,
                     "support": int(cm.loc[lab].sum())}
    result = {"n": len(d), "accuracy": float((d["true_label"] == d["sentiment"]).mean()),
              "confusion": cm, "report": pd.DataFrame(rows).T}
    if cm.loc["neutral"].sum() == 0:  # binary dataset (e.g. Sentiment140)
        decided = d[d["sentiment"] != "neutral"]
        if len(decided):
            result["binary_accuracy"] = float((decided["true_label"] == decided["sentiment"]).mean())
            result["binary_coverage"] = len(decided) / len(d)
    return result


def top_words(texts, n=10):
    counter = Counter()
    for t in texts:
        counter.update(w for w in re.findall(r"[a-z']{3,}", t.lower())
                       if w not in STOPWORDS)
    return counter.most_common(n)


# --------------------------------------------------------------------------- #
# 5. Charts (drawn on Figure objects so the CLI and the GUI share the code)
# --------------------------------------------------------------------------- #
def draw_dashboard(fig, df, metrics, threshold):
    """Distribution, score histogram, and confusion matrix (or pie chart)."""
    axes = fig.subplots(1, 3)
    counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    bars = axes[0].bar(LABELS, counts.values, color=[COLORS[l] for l in LABELS])
    axes[0].bar_label(bars)
    axes[0].set(title="Sentiment distribution", ylabel="Number of texts")

    axes[1].hist(df["compound"], bins=20, color="#4c72b0", edgecolor="white")
    for x in (-threshold, threshold):
        axes[1].axvline(x, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="VADER compound scores", xlabel="Compound (-1 to +1)",
                ylabel="Number of texts")

    if metrics:
        cm = metrics["confusion"]
        axes[2].imshow(cm.values, cmap="Blues")
        axes[2].set_xticks(range(3), LABELS)
        axes[2].set_yticks(range(3), LABELS)
        for i in range(3):
            for j in range(3):
                v = cm.values[i, j]
                axes[2].text(j, i, v, ha="center", va="center",
                             color="white" if v > cm.values.max() / 2 else "black")
        axes[2].set(title=f"Confusion matrix (accuracy {metrics['accuracy']:.0%})",
                    xlabel="VADER prediction", ylabel="Dataset label")
    else:
        nz = counts[counts > 0]
        axes[2].pie(nz.values, labels=nz.index, autopct="%1.0f%%",
                    colors=[COLORS[l] for l in nz.index])
        axes[2].set_title("Share of each sentiment")
    fig.tight_layout()
    return fig


def draw_top_words(fig, df):
    """Most common words in positive vs negative texts."""
    axs = fig.subplots(1, 2)
    for ax, lab in zip(axs, ["positive", "negative"]):
        words = top_words(df.loc[df["sentiment"] == lab, "clean_text"])
        if words:
            w, c = zip(*words[::-1])
            ax.barh(w, c, color=COLORS[lab])
        ax.set_title(f"Top words in {lab} texts")
    fig.tight_layout()
    return fig


def has_trend(df) -> bool:
    """True if the data has dates covering at least two different days."""
    return bool(df["date"].notna().any() and df["date"].dropna().dt.date.nunique() >= 2)


def draw_trend(fig, df):
    """Number of positive/neutral/negative texts per day."""
    d = df.dropna(subset=["date"]).copy()
    d["day"] = d["date"].dt.date
    trend = pd.crosstab(d["day"], d["sentiment"]).reindex(columns=LABELS, fill_value=0)
    ax = fig.subplots()
    for lab in LABELS:
        ax.plot(trend.index, trend[lab], marker="o", color=COLORS[lab], label=lab)
    ax.set(title="Sentiment over time", xlabel="Date", ylabel="Number of texts")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def make_charts(df, metrics, out_dir: Path, threshold: float, show: bool):
    """Save the charts as PNG files (CLI). With show=True also open windows."""
    from matplotlib.figure import Figure

    def new(size):
        if show:
            import matplotlib.pyplot as plt
            return plt.figure(figsize=size)
        return Figure(figsize=size)

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("dashboard.png", draw_dashboard(new((17, 4.8)), df, metrics, threshold)),
            ("top_words.png", draw_top_words(new((12, 4.5)), df))]
    if has_trend(df):
        jobs.append(("sentiment_trend.png", draw_trend(new((9, 4.5)), df)))
    saved = []
    for name, fig in jobs:
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        saved.append(out_dir / name)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close("all")
    return saved


# --------------------------------------------------------------------------- #
# 6. Reporting
# --------------------------------------------------------------------------- #
def print_report(df, metrics):
    total = len(df)
    print("\n" + "=" * 64)
    print(f"SENTIMENT REPORT  ({total:,} texts)")
    print("=" * 64)
    counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    for lab in LABELS:
        print(f"  {lab:<9}{counts[lab]:>8,}   ({counts[lab] / total:6.1%})")
    print(f"  average compound score: {df['compound'].mean():+.3f}")

    show_cols = ["compound", "clean_text"]
    print("\nMost positive texts:")
    for _, r in df.nlargest(3, "compound")[show_cols].iterrows():
        print(f"  {r['compound']:+.3f}  {r['clean_text'][:90]}")
    print("\nMost negative texts:")
    for _, r in df.nsmallest(3, "compound")[show_cols].iterrows():
        print(f"  {r['compound']:+.3f}  {r['clean_text'][:90]}")

    if metrics:
        print("\n" + "-" * 64)
        print(f"ACCURACY vs dataset labels ({metrics['n']:,} labelled texts)")
        print("-" * 64)
        print(f"  overall accuracy: {metrics['accuracy']:.1%}")
        if "binary_accuracy" in metrics:
            print("  (dataset has no neutral class, so VADER's 'neutral' counts as wrong)")
            print(f"  accuracy on texts VADER called pos/neg: {metrics['binary_accuracy']:.1%} "
                  f"({metrics['binary_coverage']:.0%} of texts)")
        print("\n  Confusion matrix (rows = dataset label, columns = VADER):")
        print("  " + metrics["confusion"].to_string().replace("\n", "\n  "))
        print("\n  Per-class metrics:")
        rep = metrics["report"].round(3)
        rep["support"] = rep["support"].astype(int)
        print("  " + rep.to_string().replace("\n", "\n  "))


def classify_one(analyzer, text, threshold):
    scores = analyzer.polarity_scores(clean_text(text))
    return label_from_score(scores["compound"], threshold), scores


def interactive(analyzer, threshold):
    print("Type a sentence to analyse (blank line or 'q' to quit).")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"q", "quit", "exit"}:
            break
        label, s = classify_one(analyzer, text, threshold)
        print(f"  {label.upper()}  (compound {s['compound']:+.3f} | "
              f"pos {s['pos']:.2f} neu {s['neu']:.2f} neg {s['neg']:.2f})")


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Sentiment analysis with NLTK VADER")
    ap.add_argument("--file", help="Kaggle (or any) CSV. Default: bundled sample data")
    ap.add_argument("--text", help="Analyse a single piece of text and exit")
    ap.add_argument("--interactive", action="store_true", help="Type sentences live")
    ap.add_argument("--text-col", help="Name of the text column (auto-detected)")
    ap.add_argument("--label-col", help="Name of the human-label column (auto-detected)")
    ap.add_argument("--date-col", help="Name of the date column (auto-detected)")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="Sample at most this many rows from big files (default 20000)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Compound score cutoff for pos/neg (default 0.05)")
    ap.add_argument("--out-dir", default="output", help="Folder for results and charts")
    ap.add_argument("--show", action="store_true", help="Open charts in a window")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    analyzer = get_analyzer()

    if args.text:
        label, s = classify_one(analyzer, args.text, args.threshold)
        print(f"{label.upper()}  {s}")
        return
    if args.interactive:
        interactive(analyzer, args.threshold)
        return

    path = args.file or SAMPLE_PATH
    print(f"Loading: {path}")
    try:
        data = load_dataset(path, args.text_col, args.label_col, args.date_col, args.max_rows)
    except DatasetError as e:
        sys.exit(f"Error: {e}")
    result = analyze(data, analyzer, args.threshold)
    metrics = evaluate(result)
    print_report(result, metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "sentiment_results.csv", index=False)
    charts = make_charts(result, metrics, out_dir, args.threshold, args.show)
    print(f"\nSaved results to {out_dir / 'sentiment_results.csv'}")
    for c in charts:
        print(f"Saved chart: {c}")


if __name__ == "__main__":
    main()
    _pause()
