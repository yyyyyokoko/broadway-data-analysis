"""
This file performs EDA and quality measures on the grosses dataset
"""

import pandas as pd
import sys
pd.set_option('display.max_columns', 10) # set max num of cols displayed

"""
takes in a data frame
performs preliminary EDAs
explores and reports data issues existed 
"""
def grosses_EDA(df):

    # check data types
    print("##############################################################")
    print("Taking a look at data types:",'\n')
    df.info()

    print('\n',"Data types for 'this_week_gross', 'diff_in_dollars', 'avg_ticket_price', 'percent_of_cap' and 'diff_percent_of_cap' are incorrect.")

    print("##############################################################")

    # how many missing values
    print("The number of missing values in each column:")
    print(df.isnull().sum())

    print("##############################################################")

    # how many rows of zeros
    zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00')]
    num_zero_rows = zero_rows.shape[0]
    print("The number of zero rows is %s"%num_zero_rows)
    print("##############################################################")

    # preliminary explorations on duplicated records
    print("There are %s duplicated records of data"%df[df[['show', 'this_week_gross', 'diff_in_dollars']].duplicated() == True].shape[0])
    print("##############################################################")


# function 1: missing value fraction
# takes in a data frame
# return total missing values / total number of data values
# missing values are determined based on conditions:
# either it is a NA value or

def missing_value(df):

    metrics_data = pd.DataFrame(df.isnull().sum(), columns=["missing_value"])

    # count NA value in the dataframe

    # naValues = df.loc[df.isnull().sum(axis=1) > 0,].shape[0]


    # accounts for zero rows
    total_zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00')] # gets all zero rows

    # zero grosses may be due to that fact that theaters are closed in holiday seasons
    # only consider shows that never have positive grosses in the entire data set to be missing

    num_real_zero_rows = 0

    for show in total_zero_rows.show.unique():
        if df[(df.show == show) & (df.this_week_gross != "$0.00")].shape[0] == 0:
            num_real_zero_rows = num_real_zero_rows + df[(df.show == show)].shape[0]

    #print(num_real_zero_rows)
    metrics_data.iloc[2:,0] = metrics_data.iloc[2:,0] + num_real_zero_rows

    # percentage of missing value:
    metrics_data.iloc[:,0] = metrics_data.iloc[:,0]/df.shape[0]

    return(metrics_data)


# function 2: data types:
# takes in a data frame and an ideal data type list
# returns total number of wrong type data values / total number of data values

def data_type(df,type_list):
    wrong_type_counts = []
    for i in range(0,len(type_list)):
        wrong_type = 0
        for k in df.iloc[:,i]:
            if (not pd.isnull(k)) & (type(k) != type_list[i]): # check if a value has a wrong type (excluding missing values)
                wrong_type = wrong_type +1
        wrong_type_counts.append(wrong_type)
    metrics_data = pd.DataFrame(wrong_type_counts, index = df.columns, columns = ["wrong_data_type"])
    metrics_data["wrong_data_type"] = metrics_data["wrong_data_type"]/df.shape[0]
    return metrics_data


# function 3: redundancy
# takes in a data frame
# report number of duplicated data rows (except the first occurrence) / total number of rows
def redundancy(df):

    dup = df[df[['show', 'this_week_gross', 'diff_in_dollars']].duplicated() == True]

    num_dup = dup.shape[0]

    # we don't want double count zero rows as both redundant and missing data so deduct them from duplicate counts
    num_zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00') & (df.avg_ticket_price == '$0.00')].shape[0]
    dup_non_zero = num_dup - num_zero_rows

    dup_fraction = []
    for i in range(0, df.shape[1]):
        dup_fraction.append(dup_non_zero/df.shape[0])

    # percentage of redundant rows:
    metrics_data = pd.DataFrame(dup_fraction, index=df.columns, columns=["redundancy"])
    return (metrics_data)

"""
takes in a data frame and reports its quality scores 
"""
def q_measure(df):
    type_list = [str, str, float, float, float, int, int, float, float]

    # The quality score of the broadway grosses contains 3 parts:
    #   missing value, wrong data type, and data redundancy.
    #   "Total score" take all these 3 dimensions into consideration

    quality_metric = pd.concat([missing_value(df), data_type(df, type_list),redundancy(df)], axis = 1)

    # Convert fraction to score; higher is better
    for col in quality_metric.columns:
        quality_metric[col] = round((1-quality_metric[col])*100,3)

    # Calculate total score, which it the mean of all the dimensions:
    quality_metric['Total_score'] = round(quality_metric.mean(axis = 1),3)

    min_score = min(quality_metric.Total_score)
    # Print the quality score:
    print('The data quality metrics is shown as the following:', '\n')
    print(quality_metric, '\n')
    # poor_atts = quality_metric[quality_metric["Total_score"] == min_score].index.tolist()
    # print('Attributes with lowest scores are:\n')
    # print('\n'.join(poor_atts))
    print("##############################################################")


def main(argv):
    # read in the data set
    df_raw = pd.read_csv("broadway_grosses.csv", encoding = "latin1")
    df_cleaned = pd.read_csv("grosses_cleaned.csv", encoding = "latin1")
    df_cleaned.info()

    # performs EDA on the raw data set
    grosses_EDA(df_raw)

    # calculates quality scores of the raw data set
    q_measure(df_raw)

    # calculates quality scores of the cleaned data set
    q_measure(df_cleaned)



if __name__ == "__main__":
    main(sys.argv)