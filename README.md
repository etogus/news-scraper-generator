# News Scraper & AI Article Generator

This project scrapes news articles from TASS and TV Rain, and generates new articles using LLM models (OpenAI GPT and YandexGPT).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

The project requires API keys for LLM generation. These are stored in `secret.py`, which is not included in the repository for security reasons.

**To set up your API keys:**

1. Copy the example file:
   ```bash
   cp secret.py.example secret.py
   ```

2. Edit `secret.py` and add your actual API keys:
   - **OpenAI API Key**: Get from https://platform.openai.com/api-keys
   - **Yandex API Key & Folder ID**: Get from https://cloud.yandex.ru/

### 3. Install ChromeDriver

The scrapers use Selenium with Chrome. Make sure you have:
- Google Chrome installed
- ChromeDriver installed and in your PATH

You can install ChromeDriver via:
```bash
# macOS (using Homebrew)
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/
```

## Workflow

The typical workflow for this project is:

1. **Scrape articles** from news sources (TASS and/or TV Rain)
2. **Generate articles** using AI models (OpenAI and/or Yandex)
3. **Combine data** from scraped and generated articles
4. **Analyze** the combined dataset

## Usage

### Step 1: Scraping Articles

**TASS Scraper:**
```bash
python tass-scraper.py
```

**TV Rain Scraper:**
```bash
python tvrain-scraper.py
```

Scraped articles are saved in the `data/` folder with timestamps.

### Step 2: Generating Articles with LLM

**OpenAI Generator:**
```bash
python openai-generator.py
```

**Yandex Generator:**
```bash
python yandex-generator.py
```

Generated articles are saved in the `generated-data/` folder.

### Step 3: Combining Data

After scraping and generating articles, combine them for analysis:

```bash
python combine_data.py
```

This creates combined datasets in the `combined-data/` folder and cleaned versions in the `cleaned-combined-data/` folder (with Yandex rejections removed).

### Step 4: Analysis

Analyze the cleaned combined data:

```bash
python analysis.py
```

The analysis script processes all files from `cleaned-combined-data/` and generates:
- Similarity analysis
- ROUGE scores
- Style analysis
- Token contributions
- Case studies
- Length comparisons

Analysis results are saved in the `analysis/` folder.

### Analysis Output Files

The `analysis/` folder contains the following output files:

**GPT Analysis Files:**
- `gpt_similarity_analysis.csv` - TF-IDF cosine similarity scores for each GPT-generated article (similarity to headline+lead and original article)
- `gpt_rouge_analysis.csv` - ROUGE-1, ROUGE-2, and ROUGE-L scores for each GPT-generated article
- `gpt_style_analysis.csv` - Analysis of Russian news style conventions (headline structure, date formats, etc.)
- `gpt_tfidf_case_studies.csv` - Qualitative examples of high/low similarity cases with full article text
- `gpt_token_contributions.csv` - Token-level contributions to TF-IDF similarity scores
- `gpt_similarity_by_newspaper.csv` - Average similarity scores grouped by newspaper (TASS vs TV Rain)
- `gpt_similarity_by_headline_length.csv` - Average similarity scores grouped by headline length buckets
- `gpt_overall_metrics.csv` - Summary statistics (mean, median, std, min, max, quartiles) for similarity and ROUGE metrics
- `gpt_tfidf_summary.csv` - Compact summary table with mean TF-IDF similarity scores by approach
- `gpt_rouge_summary.csv` - Compact summary table with mean ROUGE scores by approach
- `gpt_length_comparison.png` - Visualization comparing article lengths (original vs generated)

**YandexGPT Analysis Files:**
- `yandex_similarity_analysis.csv` - TF-IDF cosine similarity scores for each YandexGPT-generated article
- `yandex_rouge_analysis.csv` - ROUGE-1, ROUGE-2, and ROUGE-L scores for each YandexGPT-generated article
- `yandex_style_analysis.csv` - Analysis of Russian news style conventions
- `yandex_tfidf_case_studies.csv` - Qualitative examples of high/low similarity cases with full article text
- `yandex_token_contributions.csv` - Token-level contributions to TF-IDF similarity scores
- `yandex_similarity_by_newspaper.csv` - Average similarity scores grouped by newspaper
- `yandex_similarity_by_headline_length.csv` - Average similarity scores grouped by headline length buckets
- `yandex_overall_metrics.csv` - Summary statistics for similarity and ROUGE metrics
- `yandex_tfidf_summary.csv` - Compact summary table with mean TF-IDF similarity scores by approach
- `yandex_rouge_summary.csv` - Compact summary table with mean ROUGE scores by approach
- `yandex_length_comparison.png` - Visualization comparing article lengths (original vs generated)

**AI Evaluation Metrics:**
- `ai_prediction_scores.csv` - Correct/Incorrect prediction counts for Baseline and HLQ approaches
- `ai_performance_metrics.csv` - Precision, Recall, and F1 scores for Baseline and HLQ approaches

### Configuration

Both generators have configuration variables at the top of their files:

- `MAX_ARTICLES_TO_GENERATE`: Number of articles to process (set to `None` for all)
- `GENERATION_APPROACH`: One of `"universal"`, `"neutral"`, `"tass"`, or `"tvrain"`
- `SPECIFIC_FILE`: Process only a specific file (set to `None` to process all matching files)

Example:
```python
MAX_ARTICLES_TO_GENERATE = 10  # Process only 10 articles
GENERATION_APPROACH = "tass"   # Use TASS writing style
SPECIFIC_FILE = "tass_short_01_05_12_05.csv"  # Process only this file
```

## Project Structure

```
TassScraper/
├── data/                    # Scraped article data (CSV files)
├── generated-data/          # AI-generated articles
├── analysis/                # Analysis results
├── combined-data/           # Combined datasets (with all data)
├── cleaned-combined-data/   # Cleaned combined datasets (Yandex rejections removed)
├── tass-scraper.py         # TASS news scraper
├── tvrain-scraper.py       # TV Rain scraper
├── openai-generator.py     # OpenAI GPT article generator
├── yandex-generator.py     # Yandex GPT article generator
├── combine_data.py         # Combines scraped and generated data
├── analysis.py             # Analyzes combined datasets
├── requirements.txt        # Python dependencies
├── secret.py.example       # API keys template (copy to secret.py)
└── README.md               # This file
```
