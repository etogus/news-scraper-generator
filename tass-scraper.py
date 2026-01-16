import json
import random
import re
import time
from datetime import datetime
import os

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

os.makedirs("data", exist_ok=True)


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def get_articles_from_category(driver, category):
    base_url = 'https://tass.ru/'
    category_url = base_url + category
    print(f"Scraping URL: {category_url}")

    driver.get(category_url)
    time.sleep(5) # wait for the page to load completely

    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    except:
        print("Timeout waiting for page to load.")

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Find all article links using URL pattern matching
    # Pattern: /category/number (e.g., /politika/26118969)
    article_links = []
    article_pattern = re.compile(rf'^/{category}/\d+$')

    all_links = soup.find_all('a', href=True)
    seen_hrefs = set()
    
    for link in all_links:
        href = link.get('href', '').strip()
        href = href.split('?')[0].split('#')[0]

        if article_pattern.match(href):
            if href not in seen_hrefs:
                seen_hrefs.add(href)
                article_links.append(href)

    print(f"Found {len(article_links)} unique article links using URL pattern matching")
    return article_links[:5]


def scrape_article(driver, article_url):
    full_url = f"https://tass.ru{article_url}"
    print(f"Scraping article: {full_url}")

    try:
        driver.get(full_url)
        time.sleep(3)

        wait = WebDriverWait(driver, 10)
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        except:
            print("Timeout waiting for article content. Page structure might have changed.")

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        json_ld = None
        script_tags = soup.find_all('script', type='application/ld+json')
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'NewsArticle':
                    json_ld = data
                    break
            except:
                pass

        headline = ""
        if json_ld and json_ld.get('headline'):
            headline = json_ld.get('headline', '').strip()
        else:
            article_tag = soup.find('article')
            if article_tag:
                h1 = article_tag.find('h1')
                if h1:
                    headline = h1.get_text(strip=True)
            
            if not headline:
                main_tag = soup.find('main')
                if main_tag:
                    h1 = main_tag.find('h1')
                    if h1:
                        headline = h1.get_text(strip=True)
            
            if not headline:
                h1 = soup.find('h1')
                if h1:
                    headline = h1.get_text(strip=True)
        
        if not headline:
            print(f"No headline found for {full_url}, skipping article")
            return None

        lead = ""
        if json_ld and json_ld.get('description'):
            lead = json_ld.get('description', '').strip()
        else:
            container = soup.find('article') or soup.find('main')
            if container:
                h1 = container.find('h1')
                if h1:
                    for elem in h1.find_next_siblings():
                        if elem.name == 'p':
                            text = elem.get_text(strip=True)
                            if len(text) > 20:
                                lead = text
                                break
                        elif elem.name in ['div', 'section']:
                            p = elem.find('p')
                            if p:
                                text = p.get_text(strip=True)
                                if len(text) > 20:
                                    lead = text
                                    break
                        if lead:
                            break

        if not lead:
            print(f"No lead found for {full_url}, skipping article")
            return None

        date_text = ""
        if json_ld:
            date_text = json_ld.get('datePublished', '') or json_ld.get('dateCreated', '')
        
        if not date_text:
            time_elem = soup.find('time', datetime=True)
            if time_elem:
                date_text = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
        
        if not date_text:
            meta_date = soup.find('meta', property='article:published_time')
            if meta_date:
                date_text = meta_date.get('content', '')
        
        if not date_text:
            time_elem = soup.find('time')
            if time_elem:
                date_text = time_elem.get_text(strip=True)

        content = ""
        if json_ld and json_ld.get('articleBody'):
            content = json_ld.get('articleBody', '').strip()
        else:
            container = soup.find('article') or soup.find('main')
            if container:
                paragraphs = container.find_all('p')
                paragraph_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 30:  # filter out very short paragraphs
                        paragraph_texts.append(text)
                content = '\n'.join(paragraph_texts)
            else:
                paragraphs = soup.find_all('p')
                paragraph_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 30:
                        paragraph_texts.append(text)
                content = '\n'.join(paragraph_texts)

        if len(content) < 100:
            print(f"Content too short for {full_url}, skipping article")
            return None

        tags = []
        if json_ld and json_ld.get('keywords'):
            keywords = json_ld.get('keywords', '')
            if isinstance(keywords, str):
                tags = [k.strip() for k in keywords.split(',') if k.strip()]
            elif isinstance(keywords, list):
                tags = [str(k).strip() for k in keywords if k]
        
        if not tags:
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                keywords_str = meta_keywords.get('content', '')
                tags = [k.strip() for k in keywords_str.split(',') if k.strip()]

        if not tags:
            container = soup.find('article') or soup.find('main')
            if container:
                tag_links = container.find_all('a', href=True)
                for link in tag_links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    if text and len(text) < 50 and not text.startswith('http'):
                        if not any(nav_word in href.lower() for nav_word in ['/category/', '/section/', '/archive/', '/author/']):
                            tags.append(text)
                tags = tags[:10]

        article_data = {
            'url': full_url,
            'headline': headline,
            'lead': lead,
            'date': date_text,
            'content': content,
            'tags': ', '.join(tags),
            'metadata': str(json_ld) if json_ld else "No JSON-LD data found"
        }

        return article_data

    except Exception as e:
        print(f"Error scraping article {full_url}: {str(e)}")
        return None


def run_tass_scraper():
    categories = ['politika', 'ekonomika', 'obschestvo', 'mezhdunarodnaya-panorama']
    driver = setup_driver()
    all_articles_data = []

    try:
        for category in categories:
            print(f"\nScraping category: {category}")
            article_links = get_articles_from_category(driver, category)
            print(f"Found {len(article_links)} article links in {category}")

            valid_count = 0
            for article_link in article_links:
                article_data = scrape_article(driver, article_link)
                if article_data:
                    article_data['category'] = category
                    all_articles_data.append(article_data)
                    valid_count += 1
                    print(f"Successfully scraped: {article_data['headline']}")

                if valid_count >= 15:
                    break

                time.sleep(random.uniform(2, 4))

            print(f"Added {valid_count} valid articles from {category}")

    finally:
        driver.quit()

    # Create two dataframes - one with metadata only, one with full content
    if all_articles_data:
        df_full = pd.DataFrame(all_articles_data)
        df_short = df_full.drop(columns=['url', 'content', 'date', 'metadata', 'category', 'tags'])
        timestamp = datetime.now().strftime("%d_%m_%H_%M")
        short_filename = f'data/tass_short_{timestamp}.csv'
        df_short.to_csv(short_filename, index=False, encoding='utf-8-sig')
        print(f"\nSaved headline-lead only files for {len(all_articles_data)} articles to {short_filename}")

        full_filename = f'data/tass_fulltext_{timestamp}.csv'
        df_full.to_csv(full_filename, index=False, encoding='utf-8-sig')
        print(f"Saved full content for {len(all_articles_data)} articles to {full_filename}")
    else:
        print("No articles were scraped.")


if __name__ == "__main__":
    run_tass_scraper()