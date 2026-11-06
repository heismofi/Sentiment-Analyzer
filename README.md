# Sentiment Analysis Tool (NLTK + VADER)

Two ways to use it: a **desktop app (GUI)** and a **command-line tool**.

Classifies comments, tweets or reviews as **positive**, **negative** or **neutral**,
reports statistics, checks accuracy against human labels, and saves charts.

## Setup
```
pip install -r requirements.txt
```
The VADER lexicon downloads automatically on first run.

## Run
```
python sentiment_tool.py                         # bundled sample data (60 social posts)
python sentiment_tool.py --file path/to/kaggle.csv
python sentiment_tool.py --text "I love this phone!"
python sentiment_tool.py --interactive
```
Results and charts are saved to `output/` (`sentiment_results.csv`, `dashboard.png`,
`top_words.png`, `sentiment_trend.png`).

## Using a Kaggle dataset
Download one of these (free Kaggle account needed) and pass it with `--file`:
- **Sentiment140** (`training.1600000.processed.noemoticon.csv`)
- **Twitter Entity Sentiment Analysis** (`twitter_training.csv`)
- **Twitter US Airline Sentiment** (`Tweets.csv`)

Formats are auto-detected. For any other CSV the tool looks for a text column
(`text`, `tweet`, `review`...) and a label column; override with `--text-col` and `--label-col`.
Big files are sampled to 20,000 rows (`--max-rows` to change).

## How it works
1. **Load**: read the CSV, standardise columns, drop duplicates.
2. **Clean**: remove links, @mentions, `#`, HTML entities. Punctuation, CAPS and emoticons are kept because VADER uses them.
3. **Score**: VADER gives `neg / neu / pos` and a `compound` score from -1 to +1.
4. **Classify**: compound >= 0.05 positive, <= -0.05 negative, else neutral (`--threshold` to change).
5. **Evaluate**: if the dataset has human labels, show accuracy, confusion matrix, precision/recall/F1.
6. **Visualise**: distribution, score histogram, confusion matrix, top words, trend over time.

## Known limitations (good to discuss in your write-up)
- VADER is lexicon-based, so **sarcasm** ("Yeah great, another delayed delivery") is often misread.
- Slang, Pidgin and misspellings not in the lexicon are ignored.
- Sentiment140 has no neutral class, so VADER's neutral predictions count as errors; the tool also prints accuracy on pos/neg-only predictions.


## Desktop app (GUI)
```
python sentiment_gui.py        (or double-click run_gui.bat)
```
- **Analyze Text**: paste a comment and see its label, score and confidence bars.
- **Analyze Dataset**: open a Kaggle CSV (or use the sample data), then browse the summary,
  a sortable/filterable table and charts. Export results as CSV and charts as PNG.

## Make a single Windows application (.exe)
Double-click **`build_exe.bat`** (needs Python 3.9+ installed and internet access once).
It creates a clean environment, installs the dependencies, bundles the VADER lexicon and the
sample data, and produces **`dist\SentimentAnalyzer.exe`**, one file that runs on any Windows PC
with no Python and no internet.

Notes
- The first launch of the .exe takes a few seconds while it unpacks itself.
- The file is large (roughly 100-150 MB) because it contains Python, pandas and matplotlib.
- Some antivirus programs flag unsigned PyInstaller apps as suspicious. If that happens, it is a
  false positive; you can build it yourself from this source.
- To build for macOS/Linux, run the same PyInstaller command from `build_exe.bat` in a terminal
  (use `:` instead of `;` in the `--add-data` arguments).
