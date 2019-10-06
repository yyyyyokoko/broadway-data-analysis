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
    # count NA value in the dataframe
    naValues = df.loc[df.isnull().sum(axis=1) > 0,].shape[0]


    # accounts for zero rows
    total_zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00')] # gets all zero rows

    # zero grosses may be due to that fact that theathers are closed in holiday seasons
    # only consider shows that never have positive grosses in the entire data set to be missing

    num_real_zero_rows = 0

    for show in total_zero_rows.show.unique():
        if df[(df.show == show) & (df.this_week_gross != "$0.00")].shape[0] == 0:
            num_real_zero_rows = num_real_zero_rows + df[(df.show == show)].shape[0]

    # print(num_real_zero_rows)

    num_zero_values = num_real_zero_rows * (df.shape[1]-2) # calculate the total number of zero values (excluding titles and dates)

    # total numbers of value
    total_number = df.shape[0] * df.shape[1]
    # percentage of missing value:
    p_of_missing_value = (naValues + num_zero_values)/total_number
    return(p_of_missing_value)


# function 2: data types:
# takes in a data frame and an ideal data type list
# returns total number of wrong type data values / total number of data values

def data_type(df,type_list):
    wrong_type = 0
    for i in range(0,len(type_list)):
        for k in df.iloc[:,i]:
            if type(k) != type_list[i]:
                wrong_type = wrong_type +1
    # total number of values
    total_number = df.shape[0] * df.shape[1]
    # percentage of missing value:
    p_of_wrong_type = wrong_type / total_number
    return(p_of_wrong_type)


# function 3: redundancy
# takes in a data frame
# report number of duplicated data rows (except the first occurrence) / total number of rows
def redundancy(df):

    dup = df[df[['show', 'this_week_gross', 'diff_in_dollars']].duplicated() == True]

    num_dup = dup.shape[0]

    # we don't want double count zero rows as both redundant and missing data so deduct them from duplicate counts
    num_zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00') & (df.avg_ticket_price == '$0.00')].shape[0]

    dup_non_zero = num_dup - num_zero_rows

    # total number of rows
    total_number = df.shape[0]
    # percentage of redundant rows:
    p_of_redundancy = dup_non_zero / total_number
    return (p_of_redundancy)



def main(argv):
    # read in the data set
    df = pd.read_csv("broadway_grosses.csv", encoding = "latin1")

    # performs EDA on the data set
    grosses_EDA(df)

    # The quality score of the broadway grosses contains 3 parts:
    #   missing value, wrong data type, and data redundancy.
    #   "Total score" take all these 3 dimensions into consideration

    quality_metric = pd.DataFrame(columns = ['Total_score','missing_value','data_type','redundancy'])
    # 1. missing value percentage:
    #       the number of missing value / number of all the values in the dataframe
    quality_metric.loc[0,'missing_value'] = round((1-missing_value(df))*100,2)

    # 2. wrong data_type percentage:
    # if the type of a value is not the same as what we expected, it would be considered
    # as a wrong type of data
    type_list = [str, str, float, float, float, int, int, float, float]
    quality_metric.loc[0,'data_type'] = round((1 - data_type(df,type_list))*100,2)

    # 3. redundancy percentage:

    quality_metric.loc[0,'redundancy'] = round((1 - redundancy(df))*100,2)

    # Calculate total score, which it the mean of all the dimensions:
    quality_metric.loc[0,'Total_score'] = quality_metric.iloc[0,1:4].mean().round()

    # Print the quality score:
    print('The data quality metrics is shown as the following:','\n')
    print(quality_metric,'\n')
    print('The data quality score before cleaning is: ',quality_metric.loc[0,'Total_score'],'\n')

if __name__ == "__main__":
    main(sys.argv)