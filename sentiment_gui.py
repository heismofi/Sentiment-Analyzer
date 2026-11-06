#!/usr/bin/env python3
"""
Sentiment Analyzer - GUI
========================
A desktop interface for the NLTK + VADER sentiment tool.

  Tab 1  Analyze Text     type or paste a comment/tweet/review, get an instant result
  Tab 2  Analyze Dataset  load a Kaggle CSV (or the bundled sample), see the summary,
                          a sortable results table and charts, then export everything

Run:  python sentiment_gui.py        (or double-click run_gui.bat on Windows)
Needs: pip install nltk pandas matplotlib   (Tkinter ships with Python)
"""
from __future__ import annotations

import io
import queue
import re
import sys
import threading
import tkinter as tk
from contextlib import redirect_stdout
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# pythonw.exe has no console, so stdout/stderr can be None
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# Import the analysis engine; show a friendly dialog if packages are missing.
try:
    import matplotlib
    matplotlib.use("Agg")  # charts are embedded manually, never opened as pop-ups
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import sentiment_tool as st
except ImportError as exc:  # pragma: no cover
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "Missing package",
        f"{exc}\n\nInstall the required packages by opening Command Prompt and running:\n\n"
        "    pip install nltk pandas matplotlib",
    )
    sys.exit(1)

sys.excepthook = sys.__excepthook__  # the CLI's "press Enter" crash hook isn't needed here

EXAMPLES = [
    "I love this phone, the battery lasts all day!",
    "Worst customer service ever. Waited two hours and nobody helped me.",
    "The meeting is scheduled for 3pm in the main hall.",
    "Not bad at all, actually pretty good :)",
    "This update ruined the app. So frustrating!!!",
    "Yeah great, another delayed delivery. Thanks a lot...",
]
TABLE_LIMIT = 3000
ACCENT = "#1f4e79"


def bmp_only(text: str) -> str:
    """Tk can choke on emoji outside the Basic Multilingual Plane; drop them for display."""
    return re.sub("[\U00010000-\U0010FFFF]", "", str(text))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analyzer  |  NLTK + VADER")
        self.geometry("1150x780")
        self.minsize(960, 640)

        self.analyzer = None
        self.busy = False
        self.dataset_path: Path = st.SAMPLE_PATH
        self.result_df = None
        self.metrics = None
        self.last_threshold = 0.05
        self.example_index = 0
        self.queue: queue.Queue = queue.Queue()

        self.threshold_var = tk.StringVar(value="0.05")
        self.status_var = tk.StringVar(value="Loading VADER lexicon...")
        self.file_var = tk.StringVar(value="Using: bundled sample data (60 social media posts)")

        self._setup_style()
        self._build_ui()
        self._load_analyzer_async()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------ style
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=ACCENT)
        style.configure("Sub.TLabel", foreground="#555555")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=24)
        for key, color in (("pos", st.COLORS["positive"]),
                           ("neu", st.COLORS["neutral"]),
                           ("neg", st.COLORS["negative"])):
            style.configure(f"{key}.Horizontal.TProgressbar", background=color,
                            troughcolor="#e6e6e6", thickness=18)

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        header = ttk.Frame(self, padding=(14, 10, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Sentiment Analyzer", style="Title.TLabel").pack(side="left")
        right = ttk.Frame(header)
        right.pack(side="right")
        ttk.Label(right, text="Neutral threshold (±):").pack(side="left")
        ttk.Spinbox(right, from_=0.0, to=0.5, increment=0.01, width=6,
                    textvariable=self.threshold_var).pack(side="left", padx=(4, 0))

        status = ttk.Label(self, textvariable=self.status_var, relief="sunken",
                           anchor="w", padding=(8, 3))
        status.pack(side="bottom", fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self.text_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text="  Analyze Text  ")
        self.notebook.add(self.data_tab, text="  Analyze Dataset  ")
        self._build_text_tab(self.text_tab)
        self._build_data_tab(self.data_tab)

    # ------------------------------------------------------------ text tab
    def _build_text_tab(self, parent):
        frm = ttk.Frame(parent, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Enter a comment, tweet or review:",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.input_box = tk.Text(frm, height=7, wrap="word", font=("Segoe UI", 12),
                                 relief="solid", borderwidth=1, padx=8, pady=6)
        self.input_box.pack(fill="x", pady=(6, 8))
        self.input_box.bind("<Control-Return>", self._on_ctrl_enter)

        buttons = ttk.Frame(frm)
        buttons.pack(fill="x")
        self.analyze_btn = ttk.Button(buttons, text="Analyze  (Ctrl+Enter)",
                                      style="Accent.TButton", command=self.analyze_text,
                                      state="disabled")
        self.analyze_btn.pack(side="left")
        ttk.Button(buttons, text="Try an example", command=self.load_example).pack(side="left", padx=8)
        ttk.Button(buttons, text="Clear", command=self.clear_text).pack(side="left")

        result = ttk.LabelFrame(frm, text=" Result ", padding=16)
        result.pack(fill="x", pady=(18, 0))
        self.result_label = tk.Label(result, text="-", font=("Segoe UI", 30, "bold"), fg="#777777")
        self.result_label.pack(anchor="w")
        self.score_label = ttk.Label(result, text="Type some text and press Analyze.",
                                     style="Sub.TLabel")
        self.score_label.pack(anchor="w", pady=(0, 12))

        self.bars = {}
        for key, name in (("pos", "Positive"), ("neu", "Neutral"), ("neg", "Negative")):
            row = ttk.Frame(result)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=name, width=10).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, length=520,
                                  style=f"{key}.Horizontal.TProgressbar")
            bar.pack(side="left", fill="x", expand=True, padx=8)
            pct = ttk.Label(row, text="0%", width=6)
            pct.pack(side="left")
            self.bars[key] = (bar, pct)

        ttk.Label(frm, style="Sub.TLabel", wraplength=900, text=(
            "Tip: VADER reads punctuation, CAPITAL LETTERS and emoticons like :) as sentiment "
            "signals. It can misread sarcasm, e.g. 'Yeah great, another delay'.")
        ).pack(anchor="w", pady=(14, 0))

    def _on_ctrl_enter(self, _event):
        self.analyze_text()
        return "break"

    def load_example(self):
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", EXAMPLES[self.example_index % len(EXAMPLES)])
        self.example_index += 1
        if self.analyzer:
            self.analyze_text()

    def clear_text(self):
        self.input_box.delete("1.0", "end")
        self.result_label.config(text="-", fg="#777777")
        self.score_label.config(text="Type some text and press Analyze.")
        for bar, pct in self.bars.values():
            bar["value"] = 0
            pct.config(text="0%")

    def analyze_text(self):
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            self.status_var.set("Type or paste some text first.")
            return
        if self.analyzer is None:
            self.status_var.set("Still loading the VADER lexicon, try again in a moment.")
            return
        threshold = self._threshold()
        label, scores = st.classify_one(self.analyzer, text, threshold)
        self.result_label.config(text=label.upper(), fg=st.COLORS[label])
        self.score_label.config(
            text=f"Compound score: {scores['compound']:+.3f}    "
                 f"(positive >= +{threshold:g}, negative <= -{threshold:g})")
        for key, (bar, pct) in self.bars.items():
            value = scores[key] * 100
            bar["value"] = value
            pct.config(text=f"{value:.0f}%")
        self.status_var.set("Done.")

    # ------------------------------------------------------------ data tab
    def _build_data_tab(self, parent):
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        row1 = ttk.Frame(frm)
        row1.pack(fill="x")
        ttk.Button(row1, text="Open CSV file...", command=self.open_csv).pack(side="left")
        ttk.Button(row1, text="Use sample data", command=self.use_sample).pack(side="left", padx=8)
        ttk.Label(row1, textvariable=self.file_var, style="Sub.TLabel").pack(side="left", padx=8)

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=(8, 0))
        self.text_col_var = tk.StringVar()
        self.label_col_var = tk.StringVar()
        self.max_rows_var = tk.StringVar(value="20000")
        ttk.Label(row2, text="Text column:").pack(side="left")
        ttk.Entry(row2, textvariable=self.text_col_var, width=14).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Label column:").pack(side="left")
        ttk.Entry(row2, textvariable=self.label_col_var, width=14).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Max rows:").pack(side="left")
        ttk.Spinbox(row2, from_=100, to=1000000, increment=1000, width=9,
                    textvariable=self.max_rows_var).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="(leave columns blank to auto-detect)",
                  style="Sub.TLabel").pack(side="left")

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=(10, 8))
        self.run_btn = ttk.Button(row3, text="Run analysis", style="Accent.TButton",
                                  command=self.run_dataset, state="disabled")
        self.run_btn.pack(side="left")
        self.export_btn = ttk.Button(row3, text="Export results CSV...",
                                     command=self.export_csv, state="disabled")
        self.export_btn.pack(side="left", padx=8)
        self.charts_btn = ttk.Button(row3, text="Save charts as PNG...",
                                     command=self.save_charts, state="disabled")
        self.charts_btn.pack(side="left")
        self.progress = ttk.Progressbar(row3, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        self.results_nb = ttk.Notebook(frm)
        self.results_nb.pack(fill="both", expand=True)

        # Summary
        summary_tab = ttk.Frame(self.results_nb)
        self.results_nb.add(summary_tab, text=" Summary ")
        self.summary_box = ScrolledText(summary_tab, font=("Consolas", 10), wrap="none",
                                        relief="flat", padx=10, pady=8)
        self.summary_box.pack(fill="both", expand=True)
        self.summary_box.insert("1.0", "Choose a dataset (or keep the sample data) and click "
                                       "'Run analysis'.")
        self.summary_box.config(state="disabled")

        # Table
        table_tab = ttk.Frame(self.results_nb)
        self.results_nb.add(table_tab, text=" Table ")
        bar = ttk.Frame(table_tab, padding=(6, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Show:").pack(side="left")
        self.filter_var = tk.StringVar(value="all")
        filt = ttk.Combobox(bar, textvariable=self.filter_var, width=10, state="readonly",
                            values=["all", "positive", "neutral", "negative"])
        filt.pack(side="left", padx=6)
        filt.bind("<<ComboboxSelected>>", lambda _e: self._fill_table())
        self.table_info = ttk.Label(bar, text="", style="Sub.TLabel")
        self.table_info.pack(side="left", padx=10)

        holder = ttk.Frame(table_tab)
        holder.pack(fill="both", expand=True)
        columns = ("sentiment", "compound", "dataset_label", "text")
        self.tree = ttk.Treeview(holder, columns=columns, show="headings")
        widths = {"sentiment": 90, "compound": 80, "dataset_label": 100, "text": 760}
        headings = {"sentiment": "VADER", "compound": "Score",
                    "dataset_label": "Dataset label", "text": "Text"}
        for col in columns:
            self.tree.heading(col, text=headings[col],
                              command=lambda c=col: self._sort_table(c, False))
            self.tree.column(col, width=widths[col], anchor="w", stretch=(col == "text"))
        for lab in st.LABELS:
            self.tree.tag_configure(lab, foreground=st.COLORS[lab])
        ysb = ttk.Scrollbar(holder, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        # Charts
        charts_tab = ttk.Frame(self.results_nb)
        self.results_nb.add(charts_tab, text=" Charts ")
        self.chart_nb = ttk.Notebook(charts_tab)
        self.chart_nb.pack(fill="both", expand=True)

    # -------------------------------------------------- dataset actions
    def open_csv(self):
        path = filedialog.askopenfilename(
            title="Choose a CSV file (e.g. a Kaggle Twitter sentiment dataset)",
            filetypes=[("Data files (CSV, ZIP from Kaggle)", "*.csv *.zip *.tsv *.txt"), ("All files", "*.*")])
        if path:
            self.dataset_path = Path(path)
            self.file_var.set(f"Using: {self.dataset_path.name}")
            self.status_var.set("File selected. Click 'Run analysis'.")

    def use_sample(self):
        self.dataset_path = st.SAMPLE_PATH
        self.file_var.set("Using: bundled sample data (60 social media posts)")
        self.status_var.set("Sample data selected. Click 'Run analysis'.")

    def run_dataset(self):
        if self.busy:
            return
        if self.analyzer is None:
            self.status_var.set("Still loading the VADER lexicon, try again in a moment.")
            return
        try:
            max_rows = max(100, int(self.max_rows_var.get()))
        except ValueError:
            max_rows = 20000
            self.max_rows_var.set("20000")
        threshold = self._threshold()
        args = (self.dataset_path, self.text_col_var.get().strip() or None,
                self.label_col_var.get().strip() or None, max_rows, threshold)
        self.busy = True
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Analyzing... large files can take a little while.")
        threading.Thread(target=self._dataset_worker, args=args, daemon=True).start()

    def _dataset_worker(self, path, text_col, label_col, max_rows, threshold):
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                data = st.load_dataset(path, text_col=text_col, label_col=label_col,
                                       max_rows=max_rows)
                result = st.analyze(data, self.analyzer, threshold)
                metrics = st.evaluate(result)
                st.print_report(result, metrics)
            self.queue.put(("done", result, metrics, buffer.getvalue(), threshold))
        except SystemExit as exc:  # load_dataset exits with a message on bad input
            self.queue.put(("error", str(exc.code)))
        except st.DatasetError as exc:  # bad file / missing column: show the plain message
            self.queue.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _show_results(self, result, metrics, report_text, threshold):
        self.result_df, self.metrics, self.last_threshold = result, metrics, threshold

        self.summary_box.config(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", bmp_only(report_text.strip()))
        self.summary_box.config(state="disabled")

        self.filter_var.set("all")
        self._fill_table()
        self._draw_charts()

        self.export_btn.config(state="normal")
        self.charts_btn.config(state="normal")
        self.results_nb.select(0)
        self.status_var.set(f"Finished: {len(result):,} texts analyzed.")

    def _fill_table(self):
        self.tree.delete(*self.tree.get_children())
        if self.result_df is None:
            return
        df = self.result_df
        choice = self.filter_var.get()
        if choice != "all":
            df = df[df["sentiment"] == choice]
        shown = df.head(TABLE_LIMIT)
        for _, row in shown.iterrows():
            truth = row["true_label"] if isinstance(row["true_label"], str) else ""
            self.tree.insert("", "end", tags=(row["sentiment"],),
                             values=(row["sentiment"], f"{row['compound']:+.3f}", truth,
                                     bmp_only(row["clean_text"])))
        note = f"Showing {len(shown):,} of {len(df):,} rows"
        if len(df) > TABLE_LIMIT:
            note += " (export the CSV to see everything)"
        self.table_info.config(text=note + ". Click a column heading to sort.")

    def _sort_table(self, column, reverse):
        items = [(self.tree.set(k, column), k) for k in self.tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            items.sort(key=lambda t: t[0].lower(), reverse=reverse)
        for index, (_, key) in enumerate(items):
            self.tree.move(key, "", index)
        self.tree.heading(column, command=lambda: self._sort_table(column, not reverse))

    def _draw_charts(self):
        for tab in list(self.chart_nb.tabs()):
            widget = self.nametowidget(tab)
            self.chart_nb.forget(tab)
            widget.destroy()

        df, metrics, threshold = self.result_df, self.metrics, self.last_threshold
        specs = [
            ("Overview", lambda f: st.draw_dashboard(f, df, metrics, threshold)),
            ("Top words", lambda f: st.draw_top_words(f, df)),
        ]
        if st.has_trend(df):
            specs.append(("Trend over time", lambda f: st.draw_trend(f, df)))

        for title, draw in specs:
            frame = ttk.Frame(self.chart_nb)
            fig = Figure(figsize=(10, 4.6), dpi=100)
            draw(fig)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.chart_nb.add(frame, text=f" {title} ")

    def export_csv(self):
        if self.result_df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save results", defaultextension=".csv",
            initialfile="sentiment_results.csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.result_df.to_csv(path, index=False, encoding="utf-8-sig")
            self.status_var.set(f"Results saved to {path}")

    def save_charts(self):
        if self.result_df is None:
            return
        folder = filedialog.askdirectory(title="Choose a folder for the chart images")
        if folder:
            saved = st.make_charts(self.result_df, self.metrics, Path(folder),
                                   self.last_threshold, show=False)
            self.status_var.set(f"Saved {len(saved)} charts to {folder}")
            messagebox.showinfo("Charts saved",
                                "Saved:\n" + "\n".join(p.name for p in saved) + f"\n\nin {folder}")

    # -------------------------------------------------------------- helpers
    def _threshold(self) -> float:
        try:
            value = float(self.threshold_var.get())
            if not 0.0 <= value <= 1.0:
                raise ValueError
        except ValueError:
            value = 0.05
            self.threshold_var.set("0.05")
        return value

    def _load_analyzer_async(self):
        def work():
            try:
                with redirect_stdout(io.StringIO()):
                    analyzer = st.get_analyzer()
                self.queue.put(("ready", analyzer))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("analyzer_error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                kind = message[0]
                if kind == "ready":
                    self.analyzer = message[1]
                    self.analyze_btn.config(state="normal")
                    self.run_btn.config(state="normal")
                    self.status_var.set("Ready.")
                elif kind == "analyzer_error":
                    self.status_var.set("Could not load the VADER lexicon.")
                    messagebox.showerror(
                        "VADER lexicon not available",
                        "The VADER lexicon could not be loaded or downloaded.\n\n"
                        f"{message[1]}\n\nCheck your internet connection (needed once), or run "
                        "this in Command Prompt:\n\n"
                        "    python -m nltk.downloader vader_lexicon")
                elif kind == "done":
                    self._finish_run()
                    self._show_results(*message[1:])
                elif kind == "error":
                    self._finish_run()
                    self.status_var.set("Analysis failed.")
                    messagebox.showerror("Could not analyze the dataset", message[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish_run(self):
        self.busy = False
        self.progress.stop()
        self.run_btn.config(state="normal")

    def report_callback_exception(self, exc, val, tb):  # errors inside Tk callbacks
        messagebox.showerror("Unexpected error", f"{exc.__name__}: {val}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
