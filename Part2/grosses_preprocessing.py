import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import sys


def main(argv):

    # set max num of cols displayed
    pd.set_option('display.max_columns', 10)

    # read in data sets
    gs = pd.read_csv("grosses_cleaned.csv")

    # clean data
    print("Cleaning data...")
    gs = gross_cleaning(gs)
    print()

    # prints out stats summary of the selected attributes
    print("Printing stats summary...")
    print()
    col_list = ['this_week_gross', 'avg_ticket_price', 'seats_sold', 'percent_of_cap', 'perfs']
    description(gs, col_list)

    # conduct outlier exploration and mark out outliers due to the strike
    print("Exploring outliers using visualizations...")
    gs = grosses_outlier_expo(gs)
    print()

    # conduct LOF and mark out outliers
    print("Conducting LOF to detect anomaly...")
    gs = grosses_lof(gs)

    print()

    # plot LOF outliers for Wicked grosses
    print("Plotting outliers detected...")
    plot_outliers(gs)
    print()

    # bin grosses
    print("Binning grosses...")
    gs = binning_grosses(gs)
    print()

    # bin percent_of_cap
    print("Binning percent of cap...")
    gs = binning_percent_cap(gs)
    print()

    # output preprocessed data
    print("Outputting the dataset...")
    gs.to_csv('part2cleanedGrosses.csv')
    print()
    print("Done!")
##################################################


def gross_cleaning(df):
    """
    gross_cleaning function takes in the grosses dataset,
    cleans it and returns a further cleaned dataset
    """
    gs = df
    # convert 'week_ending' from strings to datetime objects
    gs['week_ending'] = pd.to_datetime(gs['week_ending'])

    # generate year, month for future analysis
    gs['year'] = gs['week_ending'].dt.year
    gs['month'] = gs['week_ending'].dt.month

    return gs


##################################################
def description(df, col_list):
    """
    the description function takes in a dataframe and a list of column names
    and prints out a summary statistics table
    which includes mean, median and standard deviation of each input attribute
    """
    df = df[col_list]
    df_stats = pd.DataFrame()
    df_stats['mean'] = df.mean()
    df_stats['median'] = df.median()
    df_stats['sd'] = df.std()
    print(df_stats)


##################################################


def grosses_outlier_expo(df):
    """
    the grosses_outlier_expo takes in the grosses data set
    plots two sets of time series data for exploring potential outliers
    marks out the data of the strike in 2007
    and returns a modified data set
    """
    gs = df

    # Outlier detection approach 1: plot time series of grosses and average ticket price

    wicked = gs[gs['show'] == 'Wicked']
    mama = gs[gs['show'] == 'Mamma Mia!']

    # avg_ticket_price against time
    plt.plot(wicked['week_ending'] , wicked['avg_ticket_price'], label = 'Wicked')
    plt.plot(mama['week_ending'], mama['avg_ticket_price'], label='Mamma Mia!')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Weekly average ticket price in USD')
    plt.title('Weekly average ticket price over time in USD: Wicked and Mamma Mia!')
    # plt.show()
    plt.savefig('price_over_time.png')
    plt.close()

    # Weekly grosses against time
    plt.plot(wicked['week_ending'] , wicked['this_week_gross'], label = 'Wicked')
    plt.plot(mama[ 'week_ending'], mama['this_week_gross'], label = 'Mamma Mia!')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Weekly grosses in USD')
    plt.title('Weekly grosses over time in USD: Wicked and Mamma Mia!')
    # plt.show()
    plt.savefig('grosses_over_time.png')
    plt.close()

    # Mark broadway strike in the week of 11/25/2017
    gs['strike'] = 0
    gs.loc[(gs.this_week_gross == 0) & (gs.diff_in_dollars == 0), 'strike'] = -1

    return gs

##################################################


def grosses_lof(df):
    """
    grosses_lof takes in the grosses dataset and perform LOF anormaly detection algorithm on it
    returns a modified dataset with an outlier identification column
    """
    gs = df
    outliers = []
    clf = LocalOutlierFactor(n_neighbors=5, algorithm='auto', contamination=0.1, n_jobs=-1, p=2)
    list = gs['show'].unique()
    for show in list:
        data = gs.loc[gs['show'] == show].iloc[:, 2:-1]
        if len(data) >= 5:
            outlier = clf.fit_predict(data)
            outliers = np.append(outliers, outlier)
        else:
            for i in range(0,len(data)):
                outliers = np.append(outliers, 0)
    gs['outlier_detection'] = outliers

    return gs

##################################################


def plot_outliers(df):
    """
    plot_outliers takes in the grosses data with the outlier identifier column
    and plot outliers identified  on grosses and prices over time for the show Wicked
    """
    gs = df

    wicked = gs[gs['show'] == 'Wicked']
    wicked_outliers = wicked[wicked['outlier_detection'] == -1]

    # in terms of grosses
    plt.plot(wicked['week_ending'], wicked['this_week_gross'])
    plt.scatter(wicked_outliers['week_ending'], wicked_outliers['this_week_gross'], color='red')
    plt.xlabel('Time')
    plt.ylabel('Weekly grosses in USD')
    plt.title('Weekly grosses over time in USD with outliers detected by LOF')
    # plt.show()
    plt.savefig('grosses_over_time_with_outliers.png')
    plt.close()

##################################################


def binning_grosses(df):
    """
    binning_grosses takes in the grosses date set
    and bin the grosses into 5 bins
    prints out the value counts for each bin and
    returns a modified data set with a new column called gross_binRange1
    """

    gs = df

    gs[gs['this_week_gross'] != 0].this_week_gross.describe()

    # binning strategy 1:
    # 0 as the first group
    # the rest are divided based on their order of magnitude

    gross_bin1 = np.array([-1, 1, 100000, 500000, 1000000, 10000000])
    gs['gross_binRange1'] = pd.cut(gs['this_week_gross'], gross_bin1)
    print(gs['gross_binRange1'].value_counts())

    return gs
##################################################


def binning_percent_cap(df):
    """
    binning_percent_cap takes in the grosses date set
    and bin the percent_of_cap into 8 bins
    prints out the value counts for each bin and
    returns a modified data set with a new column called percent_of_cap_binRange1
    """
    gs = df
    # 0 as group 1
    # 1-50%; 50%-60%, 60%-70%, 70%-80%, 80%-90%, 90-100%, >100%
    percent_of_cap_bin1 = np.array([-1, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
    gs['percent_of_cap_binRange1'] = pd.cut(gs['percent_of_cap'], percent_of_cap_bin1)
    print(gs['percent_of_cap_binRange1'].value_counts())
    return gs

##################################################


if __name__ == "__main__":
    main(sys.argv)
