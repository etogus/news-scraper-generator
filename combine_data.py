import os
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

def get_timestamp_from_filename(filename):
    filename = filename.replace('.csv', '')
    parts = filename.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[-4:])
    return None

def get_newspaper_from_filename(filename):
    filename = filename.replace('.csv', '')
    parts = filename.split('_')
    if len(parts) >= 3 and parts[1] == 'short':
        return parts[0]
    return None

def get_generation_source(filename):
    if 'gpt_generated' in filename:
        return 'gpt'
    elif 'yandex_generated' in filename:
        return 'yandex'
    return None

def is_yandex_rejection(text):
    """Check if text contains YandexGPT rejection message"""
    if not isinstance(text, str):
        return False
    normalized = text.strip().lower()
    rejection_patterns = [
        r"в\s+интернете\s+есть\s+много\s+сайтов\s+с\s+информацией",
        r"посмотрите,?\s+что\s+нашлось\s+в\s+поиске",
        r"я\s+не\s+могу\s+обсуждать\s+эту\s+тему",
        r"давайте\s+поговорим\s+о\s+чём-нибудь\s+ещё",
    ]
    for pattern in rejection_patterns:
        if re.search(pattern, normalized):
            return True
    return False

def clean_dataframe(df, source_type):
    """Remove rows with YandexGPT rejections from the dataframe"""
    if source_type == 'yandex':
        yandex_cols = [col for col in df.columns if col.startswith('yandex_generated_')]
        if yandex_cols:
            rejection_mask = df[yandex_cols].apply(
                lambda row: any(is_yandex_rejection(str(val)) for val in row), 
                axis=1
            )
            cleaned_df = df[~rejection_mask].copy()
            print(f"Removed {rejection_mask.sum()} rejected rows from Yandex data")
            return cleaned_df
    elif source_type == 'gpt':
        yandex_cols = [col for col in df.columns if col.startswith('yandex_generated_')]
        if yandex_cols:
            rejection_mask = df[yandex_cols].apply(
                lambda row: any(is_yandex_rejection(str(val)) for val in row), 
                axis=1
            )
            cleaned_df = df[~rejection_mask].copy()
            print(f"Removed {rejection_mask.sum()} rejected rows from GPT data (based on Yandex rejections)")
            return cleaned_df

    return df.copy()

def collect_yandex_rejected_headlines(yandex_gen_files):
    """Return a set of headlines that Yandex rejected across provided generated files."""
    rejected = set()
    for gen_file in yandex_gen_files:
        try:
            gen_df = pd.read_csv(gen_file)
            if 'headline' in gen_df.columns and 'generated_article' in gen_df.columns:
                mask = gen_df['generated_article'].apply(lambda x: is_yandex_rejection(str(x)))
                if mask.any():
                    rejected.update(gen_df.loc[mask, 'headline'].astype(str).str.strip().tolist())
        except Exception as e:
            print(f"Warning: could not read {gen_file}: {e}")
    return rejected

def combine_csv_files():
    combined_dir = Path('combined-data')
    combined_dir.mkdir(exist_ok=True)

    cleaned_combined_dir = Path('cleaned-combined-data')
    cleaned_combined_dir.mkdir(exist_ok=True)

    data_dir = Path('data')
    generated_dir = Path('generated-data')
    
    print(f"Looking for short files in: {data_dir.absolute()}")
    # Get short CSV files from data directory
    short_files = [f for f in data_dir.glob('*_short_*.csv')]
    print(f"Found {len(short_files)} short files: {[f.name for f in short_files]}")
    
    # Group files by newspaper, timestamp, and generation source
    file_groups = defaultdict(lambda: {
        'short_file': None, 
        'full_file': None, 
        'gpt_files': [], 
        'yandex_files': []
    })

    for short_file in short_files:
        newspaper = get_newspaper_from_filename(short_file.name)
        timestamp = get_timestamp_from_filename(short_file.name)
        
        if newspaper and timestamp:
            key = f"{newspaper}_{timestamp}"
            file_groups[key]['short_file'] = short_file
            print(f"Grouped short file: {short_file.name} -> {key}")

    print(f"\nLooking for corresponding full files...")
    for short_file in short_files:
        newspaper = get_newspaper_from_filename(short_file.name)
        timestamp = get_timestamp_from_filename(short_file.name)
        
        if newspaper and timestamp:
            key = f"{newspaper}_{timestamp}"
            full_file_pattern = f"{newspaper}_full_{timestamp}.csv"
            full_file = data_dir / full_file_pattern
            
            if full_file.exists():
                file_groups[key]['full_file'] = full_file
                print(f"Found full file: {full_file.name} for group {key}")
            else:
                print(f"Warning: No full file found for {key}")

    print(f"\nLooking for generated files in: {generated_dir.absolute()}")
    for gen_file in generated_dir.glob('*.csv'):
        if 'gpt_generated' in gen_file.name or 'yandex_generated' in gen_file.name:
            # Format: <newspaper>_short_<timestamp>_<source>_generated_<type>_approach.csv
            parts = gen_file.name.replace('.csv', '').split('_')
            if len(parts) >= 6:
                newspaper = parts[0]
                timestamp = '_'.join(parts[2:6])  # DD_MM_HH_MM
                key = f"{newspaper}_{timestamp}"
                
                if key in file_groups:
                    source = get_generation_source(gen_file.name)
                    if source == 'gpt':
                        file_groups[key]['gpt_files'].append(gen_file)
                        print(f"Grouped GPT file: {gen_file.name} -> {key}")
                    elif source == 'yandex':
                        file_groups[key]['yandex_files'].append(gen_file)
                        print(f"Grouped Yandex file: {gen_file.name} -> {key}")

    for group_key, files in file_groups.items():
        short_file = files['short_file']
        full_file = files['full_file']
        gpt_files = files['gpt_files']
        yandex_files = files['yandex_files']
        
        if not short_file or not full_file:
            print(f"\nSkipping group {group_key}: missing short file or full file")
            print(f"Short file: {short_file}")
            print(f"Full file: {full_file}")
            continue
            
        print(f"\nProcessing group: {group_key}")
        print(f"Short file: {short_file.name}")
        print(f"Full file: {full_file.name}")
        print(f"GPT files: {[f.name for f in gpt_files]}")
        print(f"Yandex files: {[f.name for f in yandex_files]}")
        
        try:
            short_df = pd.read_csv(short_file)
            full_df = pd.read_csv(full_file)
            short_df = short_df.dropna(how='all')
            full_df = full_df.dropna(how='all')
            
            if 'headline' in short_df.columns and 'headline' in full_df.columns:
                short_df['headline'] = short_df['headline'].astype(str).str.strip()
                full_df['headline'] = full_df['headline'].astype(str).str.strip()

                short_df = short_df[short_df['headline'].notna() & (short_df['headline'] != '')]
                full_df = full_df[full_df['headline'].notna() & (full_df['headline'] != '')]

                combined_df = pd.merge(
                    short_df, 
                    full_df[['headline', 'content']], 
                    on='headline', 
                    how='inner'
                )

                combined_df = combined_df.rename(columns={'lead': 'teaser', 'content': 'original_content'})
                
                print(f"Short file shape: {short_df.shape}")
                print(f"Full file shape: {full_df.shape}")
                print(f"After merging with full file: {combined_df.shape}")

                yandex_rejected_headlines = collect_yandex_rejected_headlines(yandex_files) if yandex_files else set()

                if gpt_files:
                    print(f"\nProcessing {len(gpt_files)} GPT files...")
                    gpt_combined_df = combined_df.copy()
                    
                    for gen_file in gpt_files:
                        try:
                            gen_df = pd.read_csv(gen_file)
                            gen_df = gen_df.dropna(how='all')
                            
                            if 'headline' in gen_df.columns:
                                gen_df['headline'] = gen_df['headline'].astype(str).str.strip()
                                gen_df = gen_df[gen_df['headline'].notna() & (gen_df['headline'] != '')]

                                filename_parts = gen_file.name.replace('.csv', '').split('_')
                                if len(filename_parts) >= 6:
                                    approach_type = filename_parts[-2]  # neutral, tass, tvrain, universal
                                    column_name = f"gpt_generated_{approach_type}_approach"

                                    gpt_combined_df = pd.merge(
                                        gpt_combined_df, 
                                        gen_df[['headline', 'generated_article']], 
                                        on='headline', 
                                        how='left'
                                    )

                                    gpt_combined_df = gpt_combined_df.rename(columns={'generated_article': column_name})
                                    
                                    print(f"Added GPT column: {column_name}")
                                else:
                                    print(f"Could not extract approach type from {gen_file.name}")
                            else:
                                print(f"Warning: 'headline' column not found in {gen_file.name}")
                                
                        except Exception as e:
                            print(f"Error processing GPT file {gen_file.name}: {str(e)}")

                    if len(gpt_combined_df.columns) > 3:
                        newspaper = group_key.split('_')[0]
                        timestamp = '_'.join(group_key.split('_')[1:])
                        gpt_output_name = f"{newspaper}_combined_gpt_{timestamp}.csv"
                        gpt_output_path = combined_dir / gpt_output_name

                        gpt_column_order = ['headline', 'teaser', 'original_content']
                        gpt_generated_columns = [col for col in gpt_combined_df.columns if col.startswith('gpt_generated_')]
                        gpt_generated_columns.sort()
                        gpt_column_order.extend(gpt_generated_columns)
                        
                        final_gpt_columns = [col for col in gpt_column_order if col in gpt_combined_df.columns]
                        gpt_combined_df = gpt_combined_df[final_gpt_columns]

                        gpt_combined_df.to_csv(gpt_output_path, index=False, encoding="utf-8-sig")
                        print(f"Successfully created GPT combined file: {gpt_output_name}")

                        gpt_cleaned_df = clean_dataframe(gpt_combined_df, 'gpt')
                        if yandex_rejected_headlines:
                            before = len(gpt_cleaned_df)
                            gpt_cleaned_df = gpt_cleaned_df[~gpt_cleaned_df['headline'].isin(yandex_rejected_headlines)].copy()
                            print(f"Also removed {before - len(gpt_cleaned_df)} rows from GPT cleaned file due to Yandex rejections")
                        gpt_cleaned_output_path = cleaned_combined_dir / gpt_output_name
                        gpt_cleaned_df.to_csv(gpt_cleaned_output_path, index=False, encoding="utf-8-sig")
                        print(f"Successfully created cleaned GPT combined file: {gpt_output_name}")

                if yandex_files:
                    print(f"\nProcessing {len(yandex_files)} Yandex files...")
                    yandex_combined_df = combined_df.copy()
                    
                    for gen_file in yandex_files:
                        try:
                            gen_df = pd.read_csv(gen_file)
                            gen_df = gen_df.dropna(how='all')
                            
                            if 'headline' in gen_df.columns:
                                gen_df['headline'] = gen_df['headline'].astype(str).str.strip()
                                gen_df = gen_df[gen_df['headline'].notna() & (gen_df['headline'] != '')]

                                filename_parts = gen_file.name.replace('.csv', '').split('_')
                                if len(filename_parts) >= 6:
                                    approach_type = filename_parts[-2]
                                    column_name = f"yandex_generated_{approach_type}_approach"

                                    yandex_combined_df = pd.merge(
                                        yandex_combined_df, 
                                        gen_df[['headline', 'generated_article']], 
                                        on='headline', 
                                        how='left'
                                    )

                                    yandex_combined_df = yandex_combined_df.rename(columns={'generated_article': column_name})
                                    
                                    print(f"Added Yandex column: {column_name}")
                                else:
                                    print(f"Could not extract approach type from {gen_file.name}")
                            else:
                                print(f"Warning: 'headline' column not found in {gen_file.name}")
                                
                        except Exception as e:
                            print(f"Error processing Yandex file {gen_file.name}: {str(e)}")

                    if len(yandex_combined_df.columns) > 3:
                        newspaper = group_key.split('_')[0]
                        timestamp = '_'.join(group_key.split('_')[1:])
                        yandex_output_name = f"{newspaper}_combined_yandex_{timestamp}.csv"
                        yandex_output_path = combined_dir / yandex_output_name

                        yandex_column_order = ['headline', 'teaser', 'original_content']
                        yandex_generated_columns = [col for col in yandex_combined_df.columns if col.startswith('yandex_generated_')]
                        yandex_generated_columns.sort()
                        yandex_column_order.extend(yandex_generated_columns)
                        
                        final_yandex_columns = [col for col in yandex_column_order if col in yandex_combined_df.columns]
                        yandex_combined_df = yandex_combined_df[final_yandex_columns]

                        yandex_combined_df.to_csv(yandex_output_path, index=False, encoding="utf-8-sig")
                        print(f"Successfully created Yandex combined file: {yandex_output_name}")

                        yandex_cleaned_df = clean_dataframe(yandex_combined_df, 'yandex')
                        if yandex_rejected_headlines:
                            before_y = len(yandex_cleaned_df)
                            yandex_cleaned_df = yandex_cleaned_df[~yandex_cleaned_df['headline'].isin(yandex_rejected_headlines)].copy()
                            print(f"Also removed {before_y - len(yandex_cleaned_df)} rows from Yandex cleaned file due to Yandex rejections")
                        yandex_cleaned_output_path = cleaned_combined_dir / yandex_output_name
                        yandex_cleaned_df.to_csv(yandex_cleaned_output_path, index=False, encoding="utf-8-sig")
                        print(f"Successfully created cleaned Yandex combined file: {yandex_output_name}")
                
            else:
                print(f"Warning: 'headline' column not found in one or both files")
                print(f"Available columns in short file: {short_df.columns.tolist()}")
                print(f"Available columns in full file: {full_df.columns.tolist()}")
                
        except Exception as e:
            print(f"Error processing group {group_key}: {str(e)}")

if __name__ == "__main__":
    combine_csv_files() 