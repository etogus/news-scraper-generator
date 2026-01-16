import json
import random
import re
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = 'https://tvrain.tv'
NEWS_URL = 'https://tvrain.tv/news/'
MAX_ARTICLES_TO_SCRAPE = 5  # target number of articles
REQUEST_DELAY_SECONDS = (1.5, 3.5)  # Min/Max delay between article scrapes
# --- ---

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def close_popup(driver, retries=1):
    """Attempt to close popups, if any - using flexible approaches. Returns silently if no popup exists."""
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        popup_iframe = None
        for iframe in iframes:
            title = iframe.get_attribute('title')
            if title and 'popup' in title.lower():
                popup_iframe = iframe
                break

        if not popup_iframe:
            return

        # Popup found, try to close it
        for attempt in range(1, retries + 1):
            try:
                driver.switch_to.frame(popup_iframe)
                buttons = driver.find_elements(By.TAG_NAME, "button")
                close_button = None
                for button in buttons:
                    text = button.text.lower()
                    class_attr = button.get_attribute('class') or ''
                    if any(indicator in text for indicator in ['close', 'закрыть', '×', '✕']) or \
                       any(indicator in class_attr.lower() for indicator in ['close', 'dismiss']):
                        close_button = button
                        break
                
                if close_button:
                    close_button.click()
                    driver.switch_to.default_content()
                    print("Popup closed.")
                    return
                else:
                    driver.switch_to.default_content()
                    return
            except Exception as e:
                try:
                    driver.switch_to.default_content()
                except:
                    pass
                if attempt < retries:
                    time.sleep(0.5)
    except Exception:
        try:
            driver.switch_to.default_content()
        except:
            pass
        pass

def get_article_links_selenium(driver):
    links = set()
    try:
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Find all links and filter by URL pattern
        # TV Rain article URLs: https://tvrain.tv/news/...
        all_links = soup.find_all('a', href=True)
        news_url_pattern = re.compile(rf'^{re.escape(BASE_URL)}/news/[^/]+')
        
        for link in all_links:
            href = link.get('href', '').strip()
            if href.startswith('/news/'):
                href = BASE_URL + href
            href = href.split('?')[0].split('#')[0]
            if news_url_pattern.match(href) and href != BASE_URL + '/news/':
                links.add(href)
        
        print(f"Found {len(links)} unique article links using URL pattern matching")
    except Exception as e:
        print(f"Error finding article links: {e}")
    return list(links)

def scrape_article_selenium(driver, article_url):
    print(f"Scraping article: {article_url}")
    try:
        driver.get(article_url)
        time.sleep(1)
        close_popup(driver)
        time.sleep(1)

        wait = WebDriverWait(driver, 10)
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        except:
            print("Timeout waiting for article content to load.")

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        json_ld = None
        script_tags = soup.find_all('script', type='application/ld+json')
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'NewsArticle' or data.get('@type') == 'Article':
                    json_ld = data
                    break
            except:
                pass

        headline_text = ""
        if json_ld and json_ld.get('headline'):
            headline_text = json_ld.get('headline', '').strip()
        else:
            article_tag = soup.find('article')
            if article_tag:
                h1 = article_tag.find('h1')
                if h1:
                    headline_text = h1.get_text(strip=True)
            
            if not headline_text:
                main_tag = soup.find('main')
                if main_tag:
                    h1 = main_tag.find('h1')
                    if h1:
                        headline_text = h1.get_text(strip=True)
            
            if not headline_text:
                h1 = soup.find('h1')
                if h1:
                    headline_text = h1.get_text(strip=True)

        if not headline_text:
            print(f"Missing headline, skipping: {article_url}")
            return None

        lead_text = ""
        if json_ld and json_ld.get('description'):
            lead_text = json_ld.get('description', '').strip()
        else:
            container = soup.find('article') or soup.find('main')
            if container:
                h1 = container.find('h1')
                if h1:
                    for elem in h1.find_next_siblings():
                        if elem.name == 'p':
                            text = elem.get_text(strip=True)
                            if len(text) > 20:
                                lead_text = text
                                break
                        elif elem.name in ['div', 'section']:
                            p = elem.find('p')
                            if p:
                                text = p.get_text(strip=True)
                                if len(text) > 20:
                                    lead_text = text
                                    break
                        if lead_text:
                            break

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
                    if len(text) > 30:  # filter out short
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

        if not content or len(content) < 50:
            print(f"Content too short ({len(content)} chars), skipping: {article_url}")
            return None

        return {
            'url': article_url,
            'headline': headline_text,
            'lead': lead_text if lead_text else "N/A",
            'date': date_text if date_text else "N/A",
            'content': content,
        }

    except TimeoutException:
        print(f"Timeout waiting for article content to load: {article_url}")
        return None
    except Exception as e:
        print(f"Error scraping article {article_url}: {str(e)}")
        return None

def run_tvrain_scraper_selenium():
    driver = setup_driver()
    all_articles_data = []
    collected_urls = set()

    try:
        print(f"Navigating to {NEWS_URL}")
        driver.get(NEWS_URL)
        close_popup(driver)
        print("Collecting article links...")
        article_links = get_article_links_selenium(driver)
        print(f"Found {len(article_links)} potential article links.")

        for link in article_links:
            if len(all_articles_data) >= MAX_ARTICLES_TO_SCRAPE:
                print(f"Reached target number of articles ({MAX_ARTICLES_TO_SCRAPE}). Stopping.")
                break

            if link in collected_urls:
                continue

            sleep_time = random.uniform(REQUEST_DELAY_SECONDS[0], REQUEST_DELAY_SECONDS[1])
            print(f"Waiting for {sleep_time:.2f} seconds before scraping article...")
            time.sleep(sleep_time)

            article_data = scrape_article_selenium(driver, link)
            if article_data:
                all_articles_data.append(article_data)
                print(f"Successfully scraped: {article_data['headline']} ({len(all_articles_data)}/{MAX_ARTICLES_TO_SCRAPE})")

            collected_urls.add(link)

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
    finally:
        print("Closing browser.")
        driver.quit()

    if all_articles_data:
        df_full = pd.DataFrame(all_articles_data)
        timestamp = datetime.now().strftime("%d_%m_%H_%M")
        filename = f'data/tvrain_full_{timestamp}.csv'.format(timestamp=timestamp)
        df_full['category'] = 'news'
        df_full = df_full[['headline', 'lead', 'category', 'content', 'date', 'url']]
        df_full.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(all_articles_data)} articles to {filename}")

        df_short = df_full.drop(columns=['url', 'content', 'date', 'category'])
        short_filename = f'data/tvrain_short_{timestamp}.csv'
        df_short.to_csv(short_filename, index=False, encoding='utf-8-sig')
        print(f"\nSaved headline-lead only file for {len(all_articles_data)} articles to {short_filename}")
    else:
        print("\nNo articles were successfully scraped.")

if __name__ == "__main__":
    run_tvrain_scraper_selenium()
