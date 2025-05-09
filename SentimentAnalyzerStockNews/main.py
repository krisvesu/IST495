from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define URL and tickers
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'AMD', 'META']

# Scrape news tables
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# Parse the news data
parsed_data = []

for ticker, news_table in news_tables.items():
    rows = news_table.find_all('tr')
    for row in rows:
        if row.a and row.td:
            title = row.a.get_text()
            date_data = row.td.text.strip().split(' ')

            if len(date_data) == 1:
                date = datetime.today().strftime('%Y-%m-%d')
                time = date_data[0]
            else:
                keyword = date_data[0]
                if keyword == 'Today':
                    date = datetime.today().strftime('%Y-%m-%d')
                elif keyword == 'Yesterday':
                    date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    try:
                        date = datetime.strptime(keyword, '%b-%d-%y').strftime('%Y-%m-%d')
                    except ValueError:
                        continue  # skip invalid date formats
                time = date_data[1]

            parsed_data.append([ticker, date, time, title])

# Create DataFrame
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
df.dropna(subset=['date'], inplace=True)

# Sentiment Analysis
vader = SentimentIntensityAnalyzer()
df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Group and plot
mean_df = df.groupby(['ticker', 'date']).mean(numeric_only=True)
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()

# Plotting
plt.figure(figsize=(10,8))
mean_df.plot(kind='bar')
plt.title("Average News Sentiment by Ticker and Date")
plt.ylabel("Compound Sentiment Score")
plt.xlabel("Date")
plt.tight_layout()
plt.show()
