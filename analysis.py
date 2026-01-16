import glob
import os
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)

os.makedirs('analysis', exist_ok=True)

import numpy as np

def compute_token_contributions(original, generated, vectorizer=None):
    """
    Compute token-level contributions to cosine similarity between original and generated text.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform([original, generated]).toarray()
    vocab = vectorizer.get_feature_names_out()

    orig_vec = X[0]
    gen_vec = X[1]

    # Normalize vectors as cosine similarity does
    norm_orig = np.linalg.norm(orig_vec)
    norm_gen = np.linalg.norm(gen_vec)
    if norm_orig == 0 or norm_gen == 0:
        return pd.DataFrame(columns=["token", "weight_original", "weight_generated", "contribution"])

    orig_normed = orig_vec / norm_orig
    gen_normed = gen_vec / norm_gen

    contributions = orig_normed * gen_normed
    df = pd.DataFrame({
        "token": vocab,
        "weight_original": orig_vec,
        "weight_generated": gen_vec,
        "contribution": contributions
    })

    df = df[df["contribution"] > 0]  # keep only overlapping tokens
    df = df.sort_values("contribution", ascending=False).reset_index(drop=True)
    return df

def parse_filename(filename):
    # Format: tass_combined_gpt_01_05_12_05.csv or tass_combined_yandex_01_05_12_05.csv
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 5 and parts[1] == 'combined':
        newspaper = parts[0]
        ai_model = parts[2]  # gpt or yandex
        date = f"{parts[3]}_{parts[4]}_{parts[5]}_{parts[6]}"
        return newspaper, date, ai_model
    return None, None, None

def _load_files_from(folder_pattern: str):
    all_data = []
    csv_files = glob.glob(folder_pattern)
    for file in csv_files:
        newspaper, date, ai_model = parse_filename(file)
        if newspaper and date and ai_model:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df['newspaper'] = newspaper
            df['date'] = date
            df['ai_model'] = ai_model
            df['source_file'] = os.path.basename(file)
            all_data.append(df)
        else:
            print(f"Skipping file with unexpected format: {file}")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def load_gpt():
    return _load_files_from('cleaned-combined-data/*_combined_gpt_*.csv')

def load_yandex():
    return _load_files_from('cleaned-combined-data/*_combined_yandex_*.csv')


def get_generated_content_columns(df):
    """Get all generated content columns from the dataframe."""
    generated_cols = [col for col in df.columns if col.startswith(('gpt_generated_', 'yandex_generated_'))]
    return generated_cols

def analyze_generated_content(df, generated_cols, output_prefix=''):
    """Analyze each generated content column separately."""
    results = []
    token_contrib_by_model = {}
    
    for col in generated_cols:
        if col.startswith('gpt_generated_'):
            ai_model = 'gpt'
            approach = col.replace('gpt_generated_', '').replace('_approach', '')
        elif col.startswith('yandex_generated_'):
            ai_model = 'yandex'
            approach = col.replace('yandex_generated_', '').replace('_approach', '')
        else:
            continue

        # Calculate length for this approach
        df[f'{col}_length'] = df[col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Calculate similarity to original content
        similarities = []
        if ai_model not in token_contrib_by_model:
            token_contrib_by_model[ai_model] = []
            
        for idx, row in df.iterrows():
            if pd.notna(row[col]) and pd.notna(row['original_content']):
                original = str(row['original_content'])
                generated = str(row[col])
                headline = str(row['headline'])
                lead = str(row['lead']) if 'lead' in df.columns else str(row['teaser'])
                
                try:
                    # TF-IDF similarity
                    tfidf = TfidfVectorizer()
                    tfidf_matrix = tfidf.fit_transform([original, generated, headline + " " + lead])
                    
                    cos_sim_generated = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    cos_sim_headline = cosine_similarity(tfidf_matrix[2:3], tfidf_matrix[1:2])[0][0]
                    
                    source_file = row.get('source_file', '') if 'source_file' in row.index else ''
                    similarities.append({
                        'headline': headline,
                        'teaser': lead,
                        'newspaper': row['newspaper'],
                        'ai_model': ai_model,
                        'approach': approach,
                        'file': source_file,
                        'headline_length': len(headline.split()),
                        'generated_similarity': cos_sim_generated,
                        'generated_headline_sim': cos_sim_headline,
                        'generated_length': len(generated.split()),
                        'original_length': len(original.split()),
                        'generated_text': generated,
                        'original_text': original,
                    })

                    contrib_df = compute_token_contributions(original, generated)
                    contrib_df["headline"] = row["headline"]
                    contrib_df["approach"] = approach
                    contrib_df["ai_model"] = ai_model
                    contrib_df["original_text"] = original
                    contrib_df["generated_text"] = generated
                    source_file = row.get('source_file', '') if 'source_file' in row.index else ''
                    contrib_df["file"] = source_file
                    token_contrib_by_model[ai_model].append(contrib_df)

                except Exception as e:
                    print(f"Error processing {col} for article {idx}: {e}")
        if similarities:
            results.extend(similarities)
    
    # Save token contributions once per AI model, with all approaches combined
    for ai_model, token_contrib_frames in token_contrib_by_model.items():
        if token_contrib_frames:
            all_contribs = pd.concat(token_contrib_frames, ignore_index=True)
            prefix = output_prefix if output_prefix else f'{ai_model}_'
            all_contribs.to_csv(f'analysis/{prefix}token_contributions.csv', index=False,
                                encoding='utf-8-sig')
    
    return pd.DataFrame(results)

def _extract_top_tfidf_terms(texts, top_n=10):
    vec = TfidfVectorizer(max_features=50000)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    weights = X.toarray()
    top_terms_list = []
    for row in weights:
        idxs = row.argsort()[::-1]
        top = []
        added = set()
        for i in idxs:
            term = vocab[i]
            if term in added:
                continue
            top.append((term, float(row[i])))
            added.add(term)
            if len(top) >= top_n:
                break
        top_terms_list.append(top)
    return top_terms_list

def compute_representative_tfidf_cases(df: pd.DataFrame, similarity_df: pd.DataFrame, output_prefix: str, top_n: int = 10):
    if similarity_df is None or similarity_df.empty:
        return
    # Pick one high and one low similarity case overall
    high_row = similarity_df.sort_values('generated_similarity', ascending=False).head(1)
    low_row = similarity_df.sort_values('generated_similarity', ascending=True).head(1)
    cases = []
    for label, sel in [('high', high_row), ('low', low_row)]:
        if sel.empty:
            continue
        ai_model = sel.iloc[0]['ai_model']
        approach = sel.iloc[0]['approach']
        headline = sel.iloc[0]['headline']
        gen_col = f"{ai_model}_generated_{approach}_approach"
        match = df[df['headline'].astype(str) == str(headline)]
        if match.empty or gen_col not in df.columns:
            continue
        original_text = str(match.iloc[0]['original_content'])
        generated_text = str(match.iloc[0][gen_col])
        lead = ''
        if 'lead' in match.columns:
            lead = str(match.iloc[0].get('lead', ''))
        elif 'teaser' in match.columns:
            lead = str(match.iloc[0].get('teaser', ''))
        # Extract top tf-idf terms from each text
        top_lists = _extract_top_tfidf_terms([original_text, generated_text], top_n=top_n)
        top_orig = top_lists[0]
        top_gen = top_lists[1]
        orig_terms = [t for t, _ in top_orig]
        gen_terms = [t for t, _ in top_gen]
        shared = sorted(list(set(orig_terms) & set(gen_terms)))
        unique_orig = sorted(list(set(orig_terms) - set(gen_terms)))
        unique_gen = sorted(list(set(gen_terms) - set(orig_terms)))
        source_file = match.iloc[0].get('source_file', '') if 'source_file' in match.columns else ''
        cases.append({
            'case': label,
            'ai_model': ai_model,
            'approach': approach,
            'headline': headline,
            'lead': lead,
            'file': source_file,
            'similarity_score': float(sel.iloc[0]['generated_similarity']),
            'top_terms_original': ', '.join(f"{t}" for t, _ in top_orig),
            'top_terms_generated': ', '.join(f"{t}" for t, _ in top_gen),
            'shared_terms': ', '.join(shared),
            'unique_original_terms': ', '.join(unique_orig),
            'unique_generated_terms': ', '.join(unique_gen),
            'original_text': original_text,
            'generated_text': generated_text,
        })
    if cases:
        out_df = pd.DataFrame(cases)
        out_df.to_csv(f'analysis/{output_prefix}tfidf_case_studies.csv', index=False, encoding='utf-8-sig')

def save_token_contributions_for_headline(df: pd.DataFrame, generated_cols, output_prefix: str, headline_target: str, source_file: str | None = None):
    """For a specific headline (and optionally a specific source file), compute token contributions per approach and save one CSV.

    If source_file is provided, restrict to that exact file (basename). Otherwise, use the first file that contains the headline.
    """
    if df is None or df.empty:
        return
    df_h = df[df['headline'].astype(str) == str(headline_target)]
    if df_h.empty:
        return
    if source_file:
        df_h = df_h[df_h['source_file'] == os.path.basename(source_file)]
        if df_h.empty:
            return
    else:
        first_file = df_h.iloc[0]['source_file']
        df_h = df_h[df_h['source_file'] == first_file]

    sub = df_h
    if sub.empty:
        return
    row = sub.iloc[0]
    original_text = str(row.get('original_content', ''))
    if not original_text:
        return
    all_rows = []
    for col in generated_cols:
        if pd.isna(row.get(col)):
            continue
        gen_text = str(row[col])
        if not gen_text:
            continue
        if col.startswith('gpt_generated_'):
            ai_model = 'gpt'
            approach = col.replace('gpt_generated_', '').replace('_approach', '')
        elif col.startswith('yandex_generated_'):
            ai_model = 'yandex'
            approach = col.replace('yandex_generated_', '').replace('_approach', '')
        else:
            continue
        contrib_df = compute_token_contributions(original_text, gen_text)
        if not contrib_df.empty:
            source_file = row.get('source_file', '') if 'source_file' in row.index else ''
            contrib_df.insert(0, 'approach', approach)
            contrib_df.insert(0, 'ai_model', ai_model)
            contrib_df.insert(0, 'headline', headline_target)
            contrib_df.insert(0, 'file', source_file)
            contrib_df.insert(len(contrib_df.columns), 'original_text', original_text)
            contrib_df.insert(len(contrib_df.columns), 'generated_text', gen_text)
            all_rows.append(contrib_df)
    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        out_path = f"analysis/{output_prefix}token_contributions.csv"
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

def calculate_rouge_scores(df, generated_cols):
    """Calculate ROUGE scores for each generated content column."""
    rouge = Rouge()
    rouge_scores = []
    
    for col in generated_cols:
        if col.startswith('gpt_generated_'):
            ai_model = 'gpt'
            approach = col.replace('gpt_generated_', '').replace('_approach', '')
        elif col.startswith('yandex_generated_'):
            ai_model = 'yandex'
            approach = col.replace('yandex_generated_', '').replace('_approach', '')
        else:
            continue
            
        for idx, row in df.iterrows():
            if pd.notna(row[col]) and pd.notna(row['original_content']):
                original = str(row['original_content'])
                generated = str(row[col])
                headline = str(row['headline'])
                lead = str(row['lead']) if 'lead' in df.columns else str(row['teaser'])
                
                try:
                    generated_scores = rouge.get_scores(generated, original)[0]

                    headline_lead = headline + " " + lead
                    generated_hl_scores = rouge.get_scores(generated, headline_lead)[0]
                    original_hl_scores = rouge.get_scores(original, headline_lead)[0]
                    
                    source_file = row.get('source_file', '') if 'source_file' in row.index else ''
                    rouge_scores.append({
                        'headline': headline,
                        'newspaper': row['newspaper'],
                        'ai_model': ai_model,
                        'approach': approach,
                        'file': source_file,
                        'headline_length': len(headline.split()),
                        'generated_rouge1_f': generated_scores['rouge-1']['f'],
                        'generated_rouge2_f': generated_scores['rouge-2']['f'],
                        'generated_rougeL_f': generated_scores['rouge-l']['f'],
                        'generated_hl_rouge1_f': generated_hl_scores['rouge-1']['f'],
                        'generated_hl_rouge2_f': generated_hl_scores['rouge-2']['f'],
                        'generated_hl_rougeL_f': generated_hl_scores['rouge-l']['f'],
                        'original_hl_rouge1_f': original_hl_scores['rouge-1']['f'],
                        'original_hl_rouge2_f': original_hl_scores['rouge-2']['f'],
                        'original_hl_rougeL_f': original_hl_scores['rouge-l']['f'],
                    })
                except Exception as e:
                    print(f"Error calculating ROUGE for {col} article {idx}: {e}")
    
    return pd.DataFrame(rouge_scores)

def calculate_overall_metrics(similarity_df, rouge_df, output_prefix=''):
    """
    Calculate overall summary metrics for similarity and ROUGE scores.
    """
    overall_metrics = []

    # Process similarity metrics
    if not similarity_df.empty:
        similarity_cols = ['generated_similarity', 'generated_headline_sim']
        
        for group_cols in [['ai_model'], ['ai_model', 'approach'], ['ai_model', 'newspaper']]:
            if all(col in similarity_df.columns for col in group_cols):
                grouped = similarity_df.groupby(group_cols)
                
                for name, group in grouped:
                    if isinstance(name, tuple):
                        group_name = '_'.join(str(n) for n in name)
                    else:
                        group_name = str(name)
                    
                    for col in similarity_cols:
                        if col in group.columns:
                            values = group[col].dropna()
                            if len(values) > 0:
                                overall_metrics.append({
                                    'metric_type': 'similarity',
                                    'metric_name': col,
                                    'grouping': '_'.join(group_cols),
                                    'group_value': group_name,
                                    'count': len(values),
                                    'mean': values.mean(),
                                    'median': values.median(),
                                    'std': values.std(),
                                    'min': values.min(),
                                    'max': values.max(),
                                    'q25': values.quantile(0.25),
                                    'q75': values.quantile(0.75)
                                })
    
    # Process ROUGE metrics
    if not rouge_df.empty:
        rouge_cols = [
            'generated_rouge1_f', 'generated_rouge2_f', 'generated_rougeL_f',
            'generated_hl_rouge1_f', 'generated_hl_rouge2_f', 'generated_hl_rougeL_f',
            'original_hl_rouge1_f', 'original_hl_rouge2_f', 'original_hl_rougeL_f'
        ]
        
        for group_cols in [['ai_model'], ['ai_model', 'approach'], ['ai_model', 'newspaper']]:
            if all(col in rouge_df.columns for col in group_cols):
                grouped = rouge_df.groupby(group_cols)
                
                for name, group in grouped:
                    if isinstance(name, tuple):
                        group_name = '_'.join(str(n) for n in name)
                    else:
                        group_name = str(name)
                    
                    for col in rouge_cols:
                        if col in group.columns:
                            values = group[col].dropna()
                            if len(values) > 0:
                                overall_metrics.append({
                                    'metric_type': 'rouge',
                                    'metric_name': col,
                                    'grouping': '_'.join(group_cols),
                                    'group_value': group_name,
                                    'count': len(values),
                                    'mean': values.mean(),
                                    'median': values.median(),
                                    'std': values.std(),
                                    'min': values.min(),
                                    'max': values.max(),
                                    'q25': values.quantile(0.25),
                                    'q75': values.quantile(0.75)
                                })
    
    if overall_metrics:
        metrics_df = pd.DataFrame(overall_metrics)
        column_order = ['metric_type', 'metric_name', 'grouping', 'group_value', 
                       'count', 'mean', 'median', 'std', 'min', 'max', 'q25', 'q75']
        metrics_df = metrics_df[column_order]
        metrics_df.to_csv(f'analysis/{output_prefix}overall_metrics.csv', index=False, encoding='utf-8-sig')
        print(f"Saved overall metrics to: analysis/{output_prefix}overall_metrics.csv")
        return metrics_df
    else:
        print("No metrics to aggregate")
        return pd.DataFrame()


def create_compact_summary(similarity_df, rouge_df, output_prefix=''):
    tfidf_rows = []
    rouge_rows = []

    if not similarity_df.empty and 'ai_model' in similarity_df.columns and 'approach' in similarity_df.columns:
        grouped = similarity_df.groupby(['ai_model', 'approach'])
        
        for (ai_model, approach), group in grouped:
            if 'headline' in group.columns:
                n_unique = group['headline'].nunique()
            elif 'file' in group.columns:
                if 'headline' in group.columns:
                    n_unique = group.groupby(['file', 'headline']).ngroups
                else:
                    n_unique = group['file'].nunique()
            else:
                n_unique = len(group)
            
            row = {
                'AI_Model': ai_model.upper(),
                'Approach': approach.capitalize(),
                'N': n_unique
            }

            if 'generated_similarity' in group.columns:
                row['TF-IDF_Similarity_to_Original_Mean'] = round(group['generated_similarity'].mean(), 3)
            if 'generated_headline_sim' in group.columns:
                row['TF-IDF_Similarity_to_Headline_Mean'] = round(group['generated_headline_sim'].mean(), 3)
            
            tfidf_rows.append(row)

    if not rouge_df.empty and 'ai_model' in rouge_df.columns and 'approach' in rouge_df.columns:
        grouped = rouge_df.groupby(['ai_model', 'approach'])
        
        for (ai_model, approach), group in grouped:
            if 'headline' in group.columns:
                n_unique = group['headline'].nunique()
            elif 'file' in group.columns:
                if 'headline' in group.columns:
                    n_unique = group.groupby(['file', 'headline']).ngroups
                else:
                    n_unique = group['file'].nunique()
            else:
                n_unique = len(group)
            
            row = {
                'AI_Model': ai_model.upper(),
                'Approach': approach.capitalize(),
                'N': n_unique
            }

            if 'generated_rouge1_f' in group.columns:
                row['ROUGE-1_F_Mean'] = round(group['generated_rouge1_f'].mean(), 3)
            if 'generated_rouge2_f' in group.columns:
                row['ROUGE-2_F_Mean'] = round(group['generated_rouge2_f'].mean(), 3)
            if 'generated_rougeL_f' in group.columns:
                row['ROUGE-L_F_Mean'] = round(group['generated_rougeL_f'].mean(), 3)
            
            rouge_rows.append(row)

    tfidf_summary_df = pd.DataFrame()
    if tfidf_rows:
        tfidf_summary_df = pd.DataFrame(tfidf_rows)
        column_order = ['AI_Model', 'Approach', 'N']
        metric_cols = [col for col in tfidf_summary_df.columns if col not in column_order]
        column_order.extend(sorted(metric_cols))
        tfidf_summary_df = tfidf_summary_df[[col for col in column_order if col in tfidf_summary_df.columns]]
        tfidf_summary_df = tfidf_summary_df.sort_values(['AI_Model', 'Approach']).reset_index(drop=True)
        tfidf_summary_df.to_csv(f'analysis/{output_prefix}tfidf_summary.csv', index=False, encoding='utf-8-sig')
        print(f"Saved TF-IDF summary to: analysis/{output_prefix}tfidf_summary.csv")

    rouge_summary_df = pd.DataFrame()
    if rouge_rows:
        rouge_summary_df = pd.DataFrame(rouge_rows)
        column_order = ['AI_Model', 'Approach', 'N']
        metric_cols = [col for col in rouge_summary_df.columns if col not in column_order]
        column_order.extend(sorted(metric_cols))
        rouge_summary_df = rouge_summary_df[[col for col in column_order if col in rouge_summary_df.columns]]
        rouge_summary_df = rouge_summary_df.sort_values(['AI_Model', 'Approach']).reset_index(drop=True)
        rouge_summary_df.to_csv(f'analysis/{output_prefix}rouge_summary.csv', index=False, encoding='utf-8-sig')
        print(f"Saved ROUGE summary to: analysis/{output_prefix}rouge_summary.csv")
    
    if tfidf_rows or rouge_rows:
        return tfidf_summary_df, rouge_summary_df
    else:
        print("No data for compact summary")
        return pd.DataFrame(), pd.DataFrame()

def analyze_russian_news_style(text):
    """Analyze Russian news style conventions in the text."""
    # Location-date pattern
    location_date_pattern = r'^[А-ЯA-Z]+(\s[А-ЯA-Z]+)?(\s/[А-Яа-я]+/)?(\s[А-ЯA-Z]+)?,\s+\d+\s+[а-яА-Яa-zA-Z]+'
    has_location_date = bool(re.search(location_date_pattern, text))

    # Direct quotes
    quote_pattern1 = r'«.*?»'
    quote_pattern2 = r'".*?"'
    quotes_count = len(re.findall(quote_pattern1, text)) + len(re.findall(quote_pattern2, text))

    # Officials
    pattern_title_name = re.compile(r'(?:президент|министр|глава|директор|председатель|премьер(?:-министр)?|спикер|посол|сенатор|депутат)\s+[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+){0,2}', re.IGNORECASE)
    pattern_title_org_then_name = re.compile(r'(?:министр\s+иностранных\s+дел|глава\s+МИД)[^\.!?\n,;:]{0,100}?\s[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+){0,2}', re.IGNORECASE)

    officials_matches = []
    for m in pattern_title_name.finditer(text):
        officials_matches.append((m.start(), m.group()))
    for m in pattern_title_org_then_name.finditer(text):
        officials_matches.append((m.start(), m.group()))

    officials_matches.sort(key=lambda x: x[0])
    dedup = []
    seen_spans = set()
    for pos, span in officials_matches:
        if pos in seen_spans:
            continue
        seen_spans.add(pos)
        dedup.append(span)
    officials_count = len(dedup)

    # Sources
    sources_patterns = [
        r'об\s+этом\s+сообщ(?:ил|ила|или|ается|ают|ено)\b',
        r'сообщил(?:а|и)?\s+в\s+пресс-?служб[еы]',
        r'сообщает\s+пресс-?служба',
        r'в\s+пресс-?служб[еы]\s+[^\.,]+',
        r'говорится\s+в\s+сообщени[ие]\s+[^\.,]+',
        r'как\s+сообщил[аио]?\b',
        r'как\s+сообщается\b',
        r'как\s+отметил[аио]?\b',
        r'как\s+заявил[аио]?\b',
        r'как\s+подчеркнул[аио]?\b',
        r'по\s+данным\b',
        r'по\s+информации\b',
        r'по\s+словам\b',
        r'\bпресс-?служб[аеы]\b',
        r'\bМИД\s+России\b',
        r'Министерств[оа]\s+иностранных\s+дел(\s+Российской\s+Федерации)?',
    ]
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    sources_count = 0
    for sent in sentences:
        found = False
        for pat in sources_patterns:
            if re.search(pat, sent, flags=re.IGNORECASE):
                found = True
                break
        if found:
            sources_count += 1

    return {
        'has_location_date': has_location_date,
        'quotes_count': quotes_count,
        'officials_count': officials_count,
        'sources_count': sources_count,
    }

def analyze_style_by_approach(df, generated_cols):
    """Analyze style for each approach separately."""
    style_analysis = []
    
    for col in generated_cols:
        if col.startswith('gpt_generated_'):
            ai_model = 'gpt'
            approach = col.replace('gpt_generated_', '').replace('_approach', '')
        elif col.startswith('yandex_generated_'):
            ai_model = 'yandex'
            approach = col.replace('yandex_generated_', '').replace('_approach', '')
        else:
            continue
            
        for idx, row in df.iterrows():
            if pd.notna(row[col]) and pd.notna(row['original_content']):
                original_style = analyze_russian_news_style(str(row['original_content']))
                generated_style = analyze_russian_news_style(str(row[col]))

                source_file = row.get('source_file', '') if 'source_file' in row.index else ''
                style_analysis.append({
                    'headline': row['headline'],
                    'newspaper': row['newspaper'],
                    'ai_model': ai_model,
                    'approach': approach,
                    'file': source_file,
                    'original_location_date': original_style['has_location_date'],
                    'generated_location_date': generated_style['has_location_date'],
                    'original_quotes': original_style['quotes_count'],
                    'generated_quotes': generated_style['quotes_count'],
                    'original_officials': original_style['officials_count'],
                    'generated_officials': generated_style['officials_count'],
                    'original_sources': original_style['sources_count'],
                    'generated_sources': generated_style['sources_count'],
                })
    
    return pd.DataFrame(style_analysis)

def run_full_pipeline(df: pd.DataFrame, output_prefix: str):
    if df.empty:
        print(f"No data for {output_prefix}. Skipping.")
        return

    print(f"Number of articles: {len(df)}")
    print(f"Newspapers: {', '.join(df['newspaper'].astype(str).unique())}")
    if 'ai_model' in df.columns:
        print(f"AI Models: {', '.join(df['ai_model'].astype(str).unique())}")

    generated_cols = get_generated_content_columns(df)
    print(f"Generated content columns found: {generated_cols}")
    if not generated_cols:
        print("No generated content columns found. Skipping.")
        return

    for col in generated_cols:
        df[f'{col}_length'] = df[col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    df['original_length'] = df['original_content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['headline_length'] = df['headline'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    if 'teaser' in df.columns:
        df['lead_length'] = df['teaser'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    elif 'lead' in df.columns:
        df['lead_length'] = df['lead'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Text Length Distribution by AI Model and Approach', fontsize=16)
    for i, col in enumerate(generated_cols[:4]):
        row_idx = i // 2
        col_idx = i % 2
        approach_data = df[df[col].notna()][[col, 'original_content']].copy()
        approach_data.columns = ['generated', 'original']
        approach_data['generated_length'] = approach_data['generated'].apply(lambda x: len(str(x).split()))
        approach_data['original_length'] = approach_data['original'].apply(lambda x: len(str(x).split()))
        data_for_plot = pd.melt(approach_data[['generated_length', 'original_length']], var_name='source', value_name='length')
        sns.boxplot(data=data_for_plot, x='source', y='length', ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set_title(f'{col.replace("_", " ").title()}')
        axes[row_idx, col_idx].set_ylabel('Word Count')
    plt.tight_layout()
    plt.savefig(f'analysis/{output_prefix}length_comparison.png', dpi=300, bbox_inches='tight')

    similarity_df = analyze_generated_content(df, generated_cols, output_prefix=output_prefix)
    if similarity_df.empty:
        similarity_df = pd.DataFrame()
    if not similarity_df.empty:
        sim_out = similarity_df.drop(columns=['newspaper'], errors='ignore')
        sim_out.to_csv(f'analysis/{output_prefix}similarity_analysis.csv', index=False, encoding='utf-8-sig')
        compute_representative_tfidf_cases(df, similarity_df, output_prefix)

    rouge_df = calculate_rouge_scores(df, generated_cols)
    if rouge_df.empty:
        rouge_df = pd.DataFrame()  # Ensure it's a DataFrame, not None
    if not rouge_df.empty:
        rouge_df.to_csv(f'analysis/{output_prefix}rouge_analysis.csv', index=False, encoding='utf-8-sig')

    calculate_overall_metrics(similarity_df, rouge_df, output_prefix=output_prefix)

    create_compact_summary(similarity_df, rouge_df, output_prefix=output_prefix)

    style_df = analyze_style_by_approach(df, generated_cols)
    if not style_df.empty:
        style_df.to_csv(f'analysis/{output_prefix}style_analysis.csv', index=False, encoding='utf-8-sig')

    if not similarity_df.empty:
        similarity_df['headline_length_bucket'] = pd.cut(similarity_df['headline_length'],
                                                        bins=[0, 5, 10, 15, float('inf')],
                                                        labels=['Very Short', 'Short', 'Medium', 'Long'])
        newspaper_approach_stats = similarity_df.groupby(['newspaper', 'ai_model', 'approach']).agg({
            'generated_similarity': 'mean',
            'generated_headline_sim': 'mean'
        }).reset_index()
        newspaper_approach_stats.to_csv(f'analysis/{output_prefix}similarity_by_newspaper.csv', index=False, encoding='utf-8-sig')
        headline_length_stats = similarity_df.groupby(['headline_length_bucket', 'ai_model', 'approach']).agg({
            'generated_similarity': 'mean',
            'generated_headline_sim': 'mean'
        }).reset_index()
        headline_length_stats = headline_length_stats.dropna(subset=['generated_similarity', 'generated_headline_sim'], how='all')
        headline_length_stats.to_csv(f'analysis/{output_prefix}similarity_by_headline_length.csv', index=False, encoding='utf-8-sig')


print("\n=== GPT Analysis ===")
df_gpt = load_gpt()
run_full_pipeline(df_gpt, output_prefix='gpt_')

print("\n=== Yandex Analysis ===")
df_yandex = load_yandex()
run_full_pipeline(df_yandex, output_prefix='yandex_')

print("\n=== ANALYSIS COMPLETE ===")
print("All analysis files have been saved to the 'analysis' folder.")