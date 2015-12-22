import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

from pandas.io import data

from datetime import datetime

from __future__ import division

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year-1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = data.DataReader(stock, 'yahoo', start, end)
    
AAPL.describe()

AAPL.info()

AAPL['Adj Close'].plot(legend = True, figsize = (10,4))

AAPL['Volume'].plot(legend = True, figsize = (10, 4))

ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    
    AAPL[column_name] = pd.rolling_mean(AAPL['Adj Close'], ma)
    
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots = False,
                                                                              figsize = (10, 4))
                                                                              
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(figsize = (10, 4), legend = True, linestyle='--', marker = 'o')

sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'purple')

AAPL['Daily Return'].hist(bins = 100)

closing_df = data.DataReader(tech_list, 'yahoo', start, end)['Adj Close']

closing_df.head()

tech_rets = closing_df.pct_change()

tech_rets.head()

sns.jointplot('GOOG', 'GOOG', tech_rets, kind = 'scatter', color = 'seagreen')

sns.jointplot('GOOG', 'MSFT', tech_rets, kind = 'scatter')

tech_rets.head()

sns.pairplot(tech_rets.dropna())

returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter, color = 'purple')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)

returns_fig = sns.PairGrid(closing_df.dropna())

returns_fig.map_upper(plt.scatter, color = 'purple')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)

sns.corrplot(tech_rets.dropna(), annot = True)

sns.corrplot(closing_df, annot = True)

rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(), s = area)

plt.xlabel('Expected Returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3, rad= -0.3'))
        
sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'purple')

rets['AAPL'].quantile(0.05)

def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in xrange(1, days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        drift[x] = mu*dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price
    
GOOG.head()

start_price = 540.74

for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')

runs = 1000

simulations = np.zeros(runs)

for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]
    
q = np.percentile(simulations, 1)

plt.hist(simulations, bins = 200)

plt.figtext(0.6,0.8, s = "Start price: $%.2f" %start_price)
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price-q,))
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color = 'r')
plt.title(u"Final price distribution for Google after %s days" % days, weight = 'bold');
