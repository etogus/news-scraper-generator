import os
import re
import glob
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import nltk
from nltk.corpus import stopwords

import pandas as pd

PROJECT_ROOT = \
    "/Users/g241458/Library/CloudStorage/OneDrive-SixtGmbH&Co.AutovermietungKG/TassScraper"

GENERATED_DIR = os.path.join(PROJECT_ROOT, "generated-data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def _fallback_stopwords():
    return [
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
    ]


def _load_russian_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        return stopwords.words("russian")
    except Exception:
        return _fallback_stopwords()


RUSSIAN_STOPWORDS = set(_load_russian_stopwords())


REFUSAL_PATTERNS = [
    r"в\s+интернете\s+есть\s+много\s+сайтов\s+с\s+информацией",
    r"посмотрите,?\s+что\s+нашлось\s+в\s+поиске",
]


def is_refusal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.strip().lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def find_generated_column(columns):
    for name in columns:
        low = name.lower()
        if "generated_article" in low or ("generated" in low and ("yandex" in low or "article" in low)):
            return name
    return columns[-1]


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-zа-яё\s]", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3 and t not in RUSSIAN_STOPWORDS]
    return tokens


def analyze_files():
    csv_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*yandex*.csv")))
    if not csv_files:
        print("No Yandex CSV files found in generated-data.")
        return

    per_file_stats = []
    all_tokens_rejected = Counter()
    all_tokens_accepted = Counter()
    pair_rejected = Counter()
    pair_accepted = Counter()
    token_example_rejected = {}
    token_example_accepted = {}
    total_rows = 0
    total_rejected = 0

    per_row_records = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(csv_path)

        gen_col = find_generated_column(list(df.columns))

        text_source = []
        for _, row in df.iterrows():
            headline = str(row.get("headline", ""))
            lead = str(row.get("lead", row.get("teaser", "")))
            text_source.append((headline + " " + lead).strip())

        df["is_rejected"] = df[gen_col].apply(is_refusal)

        rows = len(df)
        rejected = int(df["is_rejected"].sum())
        total_rows += rows
        total_rejected += rejected

        per_file_stats.append(
            {
                "file": os.path.basename(csv_path),
                "rows": rows,
                "rejected": rejected,
                "rejection_rate": (rejected / rows) if rows else 0.0,
            }
        )

        for row_idx, (is_rej, text) in enumerate(zip(df["is_rejected"], text_source)):
            row_tokens = tokenize(text)
            headline_val = df.loc[row_idx, "headline"] if "headline" in df.columns else ""
            lead_val = df.loc[row_idx, "lead"] if "lead" in df.columns else df.loc[row_idx, "teaser"] if "teaser" in df.columns else ""

            per_row_records.append(
                {
                    "file": os.path.basename(csv_path),
                    "row_index": row_idx,
                    "headline": headline_val,
                    "teaser": lead_val,
                    "is_rejected": bool(is_rej),
                    "tokens": " ".join(row_tokens),
                    "token_count": len(row_tokens),
                }
            )

            for t in set(row_tokens):
                if is_rej:
                    if t not in token_example_rejected:
                        token_example_rejected[t] = {
                            "headline": headline_val,
                            "teaser": lead_val,
                            "file": os.path.basename(csv_path),
                            "row_index": row_idx,
                        }
                else:
                    if t not in token_example_accepted:
                        token_example_accepted[t] = {
                            "headline": headline_val,
                            "teaser": lead_val,
                            "file": os.path.basename(csv_path),
                            "row_index": row_idx,
                        }

            if is_rej:
                all_tokens_rejected.update(row_tokens)
                for a, b in combinations(sorted(set(row_tokens)), 2):
                    pair = (a, b)
                    pair_rejected[pair] += 1
            else:
                all_tokens_accepted.update(row_tokens)
                for a, b in combinations(sorted(set(row_tokens)), 2):
                    pair = (a, b)
                    pair_accepted[pair] += 1

    stats_df = pd.DataFrame(per_file_stats).sort_values("rejection_rate", ascending=False)
    stats_out = os.path.join(ANALYSIS_DIR, "yandex_rejection_rates.csv")
    stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

    vocab = set(all_tokens_rejected.keys()) | set(all_tokens_accepted.keys())
    V = max(len(vocab), 1)
    N_rej = sum(all_tokens_rejected.values()) + V
    N_acc = sum(all_tokens_accepted.values()) + V

    rows_list = []
    for token in vocab:
        c_rej = all_tokens_rejected.get(token, 0) + 1
        c_acc = all_tokens_accepted.get(token, 0) + 1
        log_odds = (np.log(c_rej / N_rej) - np.log(c_acc / N_acc))
        rows_list.append(
            {
                "token": token,
                "count_rejected": c_rej - 1,
                "count_accepted": c_acc - 1,
                "log_odds_rejection": log_odds,
            }
        )

    for row in rows_list:
        tok = row["token"]
        ex = token_example_rejected.get(tok) or token_example_accepted.get(tok)
        if ex:
            row["is_rejected"] = tok in token_example_rejected
            row["headline"] = ex["headline"]
            row["teaser"] = ex["teaser"]

    words_df = pd.DataFrame(rows_list).sort_values("log_odds_rejection", ascending=False)
    words_out = os.path.join(ANALYSIS_DIR, "yandex_word_log_odds.csv")
    words_df.to_csv(words_out, index=False, encoding="utf-8-sig")

    overall_rate = (total_rejected / total_rows) if total_rows else 0.0
    print(f"Analyzed {len(csv_files)} files. Total rows: {total_rows}. Rejection rate: {overall_rate:.2%}")
    print(f"Saved per-file stats to: {stats_out}")
    print(f"Saved word correlations to: {words_out}")


if __name__ == "__main__":
    analyze_files()


