import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(df):
    gs = df
    fig, axes = plt.subplots(1, 3, figsize = (10, 5))
    fig.tight_layout()
    X1 = gs['this_week_gross']
    X2 = gs['avg_ticket_price']
    X3 = gs['percent_of_cap']
    X1.hist(bins=20, ax=axes[0])
    axes[0].set_title('Weekly Grosses')
    axes[0].set_xlabel('Grosses in USD')
    X2.hist(bins=20, ax=axes[1])
    axes[1].set_title('Average Ticket Price')
    axes[1].set_xlabel('Price in USD')
    X3.hist(bins=20, ax=axes[2])
    axes[2].set_title('Percent of Cap')
    axes[2].set_xlabel('Percent of Cap')
    #plt.show()
    plt.savefig('hist.png', bbox_inches = "tight")
    #plt.clf()

def correlation_plotting(df,col_list):
    print('The correlation of '+str(col_list))
    df_correlation = df[col_list]
    print(df_correlation.corr())

    plt.figure(figsize=(15, 5))
    for i in range(0,3):
        # Create subplot
        plt.subplot(1,3,i+1)
        if i == 2:
            col2 = 0
        else:
            col2 = i+1
        x = df[col_list[i]]
        y = df[col_list[col2]]

        ## set correlation line
        par = np.polyfit(x, y, 1, full=True)
        slope = par[0][0]
        intercept = par[0][1]
        xl = [min(x), max(x)]
        yl = [slope * xx + intercept for xx in xl]

        ## plotting
        plt.scatter(x, y)
        plt.plot(xl, yl, '-r')
        titleLabel = "Correlation of"+ str(col_list[i]) +" and "+ str(col_list[col2])
        plt.title(titleLabel)
        plt.xlabel(col_list[i])
        plt.ylabel(col_list[col2])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=1, hspace=1)
    #plt.show()
    # Write to file
    fileName = 'CorrelationPlot.png'
    plt.savefig(fileName)

if __name__ == "__main__":
    # read in data set
    df = pd.read_csv('part2cleanedGrosses.csv', sep=',', encoding='latin1')
    df = df.drop('Unnamed: 0',axis= 1)

    # plot histogram
    plot_hist(df)

    ## correlation of the 3 variables and plots
    col_list = ['seats_sold', 'avg_ticket_price', 'percent_of_cap']
    correlation_plotting(df, col_list)