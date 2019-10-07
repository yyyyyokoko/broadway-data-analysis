"""
This script performs quality measures on musical_rating data,
cleans the data and measures the quality after cleaning
"""
import pandas as pd
import sys


# missing value fraction
# takes in a data frame
# return total missing values / total number of data values

def missing_value(df):

    metrics_data = pd.DataFrame(df.isnull().sum(), columns=["missing_value"])

    # count NA value in the dataframe

    # percentage of missing value:
    metrics_data.iloc[:,0] = metrics_data.iloc[:,0]/df.shape[0]

    return(metrics_data)


# function: redundancy
# takes in a data frame
# report number of duplicated data rows (except the first occurrence) / total number of rows

def redundancy(df):

    dup = df[df.duplicated() == True]

    num_dup = dup.shape[0]

    dup_fraction = []
    for i in range(0, df.shape[1]):
        dup_fraction.append(num_dup/df.shape[0])

    # percentage of redundant rows:
    metrics_data = pd.DataFrame(dup_fraction, index=df.columns, columns=["redundancy"])
    return (metrics_data)

"""
takes in a data frame and reports its quality scores 
"""

def q_measure(df):

    quality_metric = pd.concat([missing_value(df), redundancy(df)], axis=1)

    for col in quality_metric.columns:
        quality_metric[col] = round((1-quality_metric[col])*100,3)
    # Calculate total score, which it the mean of all the dimensions:
    quality_metric['Total_score'] = round(quality_metric.mean(axis = 1),3)

    print('The data quality metrics is shown as the following:', '\n')
    print(quality_metric, '\n')

    print("##############################################################")

"""
cleans missing data
"""

def ratings_cleaning(df):
    indexNames = df[(df['total_rating'] == 0) & (df['critics_rating'] == 0)].index
    df.drop(indexNames, inplace=True)
    ratings = df.drop_duplicates()
    ratings.to_csv("Musical_ratings-withoutNa-cleaned.csv", index=False)


def main(argv):
    # read in the data set
    df_raw = pd.read_csv("Musical_ratings-withoutNa.csv", encoding = "latin1")
    q_measure(df_raw)

    ratings_cleaning(df_raw)

    # calculates quality scores of the raw data set


    # calculates quality scores of the cleaned data set
    df_cleaned = pd.read_csv("Musical_ratings-withoutNa-cleaned.csv", encoding = "latin1")
    q_measure(df_cleaned)



if __name__ == "__main__":
    main(sys.argv)