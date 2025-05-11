from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Step 1: Scrape all unique tickers from FinViz news page
news_url = 'https://finviz.com/news.ashx?v=3'
req = Request(url=news_url, headers={'user-agent': 'Mozilla/5.0'})
response = urlopen(req)
html = BeautifulSoup(response, 'html.parser')

tickers = set()
for link in html.find_all('a', href=True):
    href = link['href']
    if 'quote.ashx?t=' in href:
        ticker = href.split('quote.ashx?t=')[1].split('&')[0].split('#')[0]
        if ticker:
            tickers.add(ticker.upper())
tickers = sorted(tickers)
print("Tickers found on FinViz news page:")
print(tickers)

# Step 2: Scrape news headlines for each ticker
finviz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'Mozilla/5.0'})
    try:
        response = urlopen(req)
        soup = BeautifulSoup(response, 'html.parser')
        news_table = soup.find('table', {'class': 'fullview-news-outer'})
        if news_table:
            news_tables[ticker] = news_table
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")

# Step 3: Parse news headlines
parsed_data = []
for ticker, news_table in news_tables.items():
    rows = news_table.find_all('tr')
    last_date = None
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 2:
            continue
        date_time = cols[0].text.strip()
        title = cols[1].text.strip()
        if ' ' in date_time:
            date_str, time_str = date_time.split(' ')
            try:
                date = datetime.strptime(date_str, '%b-%d-%y').date()
            except ValueError:
                date = last_date
            last_date = date
        else:
            time_str = date_time
            date = last_date
        if date is not None:
            parsed_data.append([ticker, date, time_str, title])

# Step 4: Create DataFrame
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df.dropna(subset=['date'], inplace=True)

# Step 5: Sentiment analysis
vader = SentimentIntensityAnalyzer()
df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Step 6: Group and pivot for visualization
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack(level=0)

# Step 7: Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(mean_df.transpose(), cmap='RdYlGn', center=0, annot=False, cbar_kws={'label': 'Sentiment'})
plt.title("Sentiment Heatmap by Ticker and Date", fontsize=16, fontweight='bold')
plt.ylabel("Ticker", fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.tight_layout()
plt.show()
