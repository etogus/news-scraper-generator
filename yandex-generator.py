import requests
import pandas as pd
import time
import random
import os
import re
from secret import YANDEX_API_KEY, YANDEX_FOLDER_ID

DATA_DIR = "data"
OUTPUT_SUFFIX = "_yandex_generated"
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
MODEL = "yandexgpt"
# Maximum number of articles to generate per file (set to None to process all articles)
MAX_ARTICLES_TO_GENERATE = None
# Generation approach: "universal", "neutral", "tass", "tvrain"
GENERATION_APPROACH = "universal"
# Specific file to process (set to None to process all matching files)
# Example: "tvrain_short_04_05_18_42.csv"
SPECIFIC_FILE = "tass_short_01_05_12_05.csv"

def get_prompts(headline, teaser, approach="universal"):
    """
    Get prompts based on the selected approach.
    """
    
    if approach == "universal":
        system_prompt = "Вы опытный российский журналист, работающий в новостном издании."
        user_prompt = f"""Напишите новостную статью на русском языке, основанную на следующем заголовке и тизере: 
Заголовок: {headline}
Тизер: {teaser}
Каждая статья должна: 
- Быть написана в новостном стиле
- Быть фактически согласованной с заголовком и тизером
"""
    
    elif approach == "neutral":
        system_prompt = "Вы опытный российский журналист. Ваша задача: написать новостную статью в нейтральном стиле без выражения личного мнения."
        user_prompt = f"""Заголовок: {headline}
Тизер: {teaser} 
Напишите полную новостную статью на русском языке, опираясь только на эти материалы. 
Статья должна быть фактически согласована с заголовком и тизером.
"""
    
    elif approach == "tass":
        system_prompt = "Вы опытный российский журналист, работающий в информационном агентстве ТАСС. Вы придерживаетесь официального стиля государственных СМИ: формальный тон, акцент на заявлениях официальных лиц и ведомств, использование терминологии, соответствующей государственной позиций, передача государственной повестки."
        user_prompt = f"""Заголовок: {headline} 
Тизер: {teaser} 
Напишите новостную статью в стиле ТАСС.
"""
    
    elif approach == "tvrain":
        system_prompt = "Вы опытный российский журналист, работающий в независимом СМИ «Дождь». Вы придерживаетесь аналитического и критического стиля: уточнение контекста, включение альтернативных точек зрения."
        user_prompt = f"""Заголовок: {headline} 
Тизер: {teaser} 
Напишите новостную статью в стиле телеканала «Дождь».
"""

    return system_prompt, user_prompt

def generate_article(headline, lead, approach="universal"):
    """
    Generate article using Yandex GPT with the specified prompt approach.
    """
    system_prompt, user_prompt = get_prompts(headline, lead, approach)
    
    headers = {
        "Authorization": f"Bearer {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/{MODEL}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": user_prompt
            }
        ]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        print(f"Error generating article: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def process_csv(file_path, approach="universal", max_articles=None):
    print(f"Processing: {file_path} with {approach} approach")
    df = pd.read_csv(file_path)

    if not {'headline', 'lead'}.issubset(df.columns):
        print(f"Missing columns in {file_path}")
        return

    total_articles = len(df)
    if max_articles:
        if max_articles < total_articles:
            df = df.head(max_articles)
            print(f"Limited to {max_articles} articles (file has {total_articles} articles)")
        elif max_articles > total_articles:
            print(f"Requested {max_articles} articles, but file only has {total_articles} articles. Processing all {total_articles} articles.")
        else:
            print(f"Processing all {total_articles} articles (matches requested {max_articles})")
    else:
        print(f"Processing all {total_articles} articles")

    generated_articles = []
    for i, row in df.iterrows():
        print(f"Generating article {i+1}/{len(df)}")
        article = generate_article(row['headline'], row['lead'], approach)
        generated_articles.append(article)
        time.sleep(random.uniform(1, 3)) # a small delay between requests

    df['generated_article'] = generated_articles

    base_name = os.path.basename(file_path)
    name_part, ext = os.path.splitext(base_name)
    output_filename = f"generated-data/{name_part}{OUTPUT_SUFFIX}_{approach}_approach.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_filename}")

def main():
    for filename in os.listdir(DATA_DIR):
        if re.match(r".+_short_\d{2}_\d{2}_\d{2}_\d{2}\.csv", filename):
            if SPECIFIC_FILE and filename != SPECIFIC_FILE:
                continue
            file_path = os.path.join(DATA_DIR, filename)
            process_csv(file_path, GENERATION_APPROACH, max_articles=MAX_ARTICLES_TO_GENERATE)

if __name__ == "__main__":
    main() 