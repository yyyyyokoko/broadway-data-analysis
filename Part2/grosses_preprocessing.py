import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

pd.set_option('display.max_columns', 10)  # set max num of cols displayed

gs = pd.read_csv("grosses_cleaned.csv")

# convert date from string to date
gs['week_ending'] = pd.to_datetime(gs['week_ending'])

##################################################
# Summary stats of the dataset
gs.describe()

# Attributes with potential outliers in
# this_week_gross: the mean is 5*10^5, but min is 0, max triples the 3rd quantile
# diff_in_dollars: extreme max
# avg_ticket_price: the mean is 67, max is 511
# seats_sold: extreme max
# perfs: extreme max and 0s
# percent_of_cap
# diff_percent_of_cap


# Double confirm with visualization
gs.hist()
# avg_ticket_price, this_week_gross

# Issues worth noting
# 0 values may indicate holiday seasons when theaters are closed
# percent_of_cap: greater than 100% indicates oversales
# Extreme values of diff_in_dollars and diff_percent_of_cap may be due to comparing with weeks when theaters are closed

##################################################

# plot time series
# weekly gross against time
gs.plot(x='week_ending', y='this_week_gross', figsize=(12,6))
plt.xlabel('Week')
plt.ylabel('Weekly Grosses in USD')
plt.title('Weekly Grosses over time in USD')
#
# avg_ticket_price against time
gs.plot(x='week_ending', y='avg_ticket_price', figsize=(12,6))
plt.xlabel('Week')
plt.ylabel('Weekly average ticket price in USD')
plt.title('Weekly average ticket price over time in USD');
plt.show()
##################################################
# outlier detection approach
# grosses = avg_ticket_price * seats_sold

wicked = gs[gs['show'] == 'Wicked']
mama = gs[gs['show'] == 'Mamma Mia!']

# avg_ticket_price against time
plt.plot(wicked['week_ending'] , wicked['avg_ticket_price'])
plt.plot(mama[ 'week_ending'], mama['avg_ticket_price'])
plt.xlabel('Week')
plt.ylabel('Weekly average ticket price in USD')
plt.title('Weekly average ticket price over time in USD')
plt.show()


##################################################
def lof(df):
    outliers = []
    clf = LocalOutlierFactor(n_neighbors=5, algorithm='auto', contamination=0.1, n_jobs=-1, p=2)
    list = gs['show'].unique()
    for show in list:
        data = gs.loc[gs['show'] == show].iloc[:, 2:-1]
        if len(data) >= 5:
            outlier = clf.fit_predict(data)
            outliers = np.append(outliers, outlier)
        else:
            for i in range(0, len(data)):
                outliers = np.append(outliers, 0)
    df['outlier_detection'] = outliers
    return df


##################################################
# run LOF on gs
gs = lof(gs)

##################################################

total_zero_rows = gs[(gs.this_week_gross == 0) & (gs.diff_in_dollars == 0)] # gets all zero rows

lof_outliers = gs[gs['outlier_detection'] == -1]

wicked = gs[gs['show'] == 'Wicked']
wicked_outliers = wicked[wicked['outlier_detection'] == -1]

# in terms of average ticket price
plt.plot(wicked['week_ending'], wicked['avg_ticket_price'])
plt.scatter(wicked_outliers['week_ending'], wicked_outliers['avg_ticket_price'], color='red')
plt.xlabel('Week')
plt.ylabel('Weekly average ticket price in USD')
plt.title('Weekly average ticket price over time in USD')
plt.show()

# in terms of average ticket price
plt.plot(wicked['week_ending'], wicked['this_week_gross'])
plt.scatter(wicked_outliers['week_ending'], wicked_outliers['this_week_gross'], color='red')
plt.xlabel('Week')
plt.ylabel('Weekly average ticket price in USD')
plt.title('Weekly average ticket price over time in USD')
plt.show()

##################################################
# Binning grosses

# Take a look at min and max, excluding 0 values
gs[gs['this_week_gross'] != 0].this_week_gross.describe()

# binning strategy 1:
# 0 as the first group
# the rest are divided based on their order of magnitude

gross_bin1= np.array([-1, 1, 100000, 500000, 1000000, 10000000])
gs['gross_binRange1'] = pd.cut(gs['this_week_gross'], gross_bin1)
# gs['gross_binRange1].value_counts

# gross_bin1 = [-1, 1, 100000, 500000, 1000000, 10000000]
# gross_groupNames = range(1,6)
# gross_binGroup1 = pd.cut(gs['this_week_gross'], gross_bin1, labels = gross_groupNames)

##################################################
# Binning avg_ticket_price

# Take a look at min and max, excluding 0 values
gs[gs['avg_ticket_price'] != 0].avg_ticket_price.describe()

# binning strategy 1:
# 0 as group 1
price_bin1 = np.array([-1, 1, 20, 40, 60, 80, 100, 1000])
gs['price_binRange1'] = pd.cut(gs['avg_ticket_price'], price_bin1)

##################################################
# Binning percent_of_cap

# Take a look at min and max, excluding 0 values
gs[gs['percent_of_cap'] != 0].percent_of_cap.describe()

# binning strategy 1:
# 0 as group 1
# 1-50%; 50%-60%, 60%-70%, 70%-80%, 80%-90%, 90-100%, >100%
percent_of_cap_bin1 = np.array([-1, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
gs['percent_of_cap_binRange1'] = pd.cut(gs['percent_of_cap'], percent_of_cap_bin1)

gs.to_csv('part2cleanedGrosses.csv')
