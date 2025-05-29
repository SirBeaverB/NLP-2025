import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# 配置
START_DATE = '2023-05-01'
END_DATE = '2024-05-01'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def daterange(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    for n in range((end - start).days + 1):
        yield (start + timedelta(n)).strftime('%Y-%m-%d')

def extract_title(soup):
    for selector in ['h3', 'h1', 'h2', 'title']:
        tag = soup.select_one(selector)
        if tag and tag.get_text(strip=True):
            return tag.get_text(strip=True)
    return ''

def fetch_article(date, ban, idx):
    date_str1 = date.replace('-', '/').replace('/', '-', 1)
    date_str2 = date.replace('-', '')
    url = f"https://paper.people.com.cn/rmrb/html/{date_str1}/nw.D110000renmrb_{date_str2}_{idx}-{ban:02d}.htm"
    try:
        resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        resp.encoding = 'utf-8'
        #print(f"Trying: {url} status: {resp.status_code}")
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'lxml')
        title = extract_title(soup)
        content = soup.select('div#ozoom p')
        if not title or not content:
            print(f"  No title or content at {url}")
            return None
        return {
            'title': title,
            'url': url,
            'date': date,
            'content': '\n'.join([p.get_text(strip=True) for p in content])
        }
    except Exception as e:
        print(f"  Exception at {url}: {e}")
        return None

def crawl():
    for date in tqdm(list(daterange(START_DATE, END_DATE)), desc='date crawling'):
        all_articles = []
        end_day = False
        for ban in range(1, 25):  # 版面号1~24
            for idx in range(1, 21):  # 每版最多20篇文章
                article = fetch_article(date, ban, idx)
                if article is None:
                    if idx == 1:
                        end_day = True  # 版面第一篇404，结束这一天
                    break  # idx出现404，结束当前版面
                all_articles.append(article)
            if end_day:
                break  # 结束当天
        if all_articles:
            with open(os.path.join(DATA_DIR, f'{date}.json'), 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, ensure_ascii=False, indent=2)
            # print(f'{len(all_articles)} written: {date}')
        else:
            print(f'{date} no articles')

if __name__ == '__main__':
    crawl() 