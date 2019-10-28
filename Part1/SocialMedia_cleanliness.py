from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt

def main(argv):
    df_before = pd.read_csv('scrap_of_social_media.csv', sep=',', encoding='latin1')
    metric_before = score(df_before,status = 'before')
    # Print the quality score:
    print('The data quality score before cleaning is: ', metric_before['total_score'], '\n')
    print('Total score of the data set before cleaning: ',round(metric_before['total_score'].mean(),2),'\n')
    #print('Detailed Score Metrics:')
    #print(metric_before)

    df_after = pd.read_csv('cleaned_SocialMedia.csv', sep=',', encoding='latin1')
    metric_after = score(df_after,status = 'after')
    # Print the quality score:
    print('The data quality score after cleaning is: ', metric_after['total_score'], '\n')
    print('Total score of the data set after cleaning: ',round(metric_after['total_score'].mean(),2),'\n')
    #print('Detailed Score Metrics:')
    #print(metric_after)

def score(df,status):
    # The quality score of social media data contains 3 parts:
    #   missing value, wrong data type, and data redundancy.
    #   "Total score" take all these 3 dimensions into consideration
    col_names = df.columns.tolist()
    quality_metric = pd.DataFrame(columns = ['attribute_name','total_score','missing_value','data_type','redundancy','erroneous_data'])

    quality_metric['attribute_name'] = col_names
    quality_metric = quality_metric.set_index('attribute_name')

    # 1. missing value percentage:
    #       the number of missing value / number of all the values in the dataframe
    quality_metric['missing_value'] = round((1-missing_value(df))*100,2)
    # the dataset does not contain missing value

    # 2. wrong data_type percentage:
    # if the type of a value is not the same as what we expected, it would be considered
    # as a wrong type of data
    type_list = [str,str,int,int,int,int,int,
             int,int,int,int,int,str,str,float]
    wrong_score = round((1 - data_type(df,type_list))*100,2)
    counter = 0
    for i in col_names[0:15]:
        quality_metric.loc[i,'data_type'] = wrong_score.iloc[counter,0]
        counter = counter +1

    # 3. redundancy percentage:
    # two "key" are used together to identify a record: the name of the show and date
    # if there are 2 or more than 2 records have the same key, they would be considered
    # as redundant data
    #p_of_redundancy = round((1 - redundancy(df))*100,2)
    quality_metric['redundancy'] = round((1 - redundancy(df)) * 100, 2)
   # counter = 0
    #for i in col_names:
       # quality_metric.loc[i, 'redundancy'] = p_of_redundancy
       # counter = counter + 1

    # 4. erroneous data percentage:
    # In the following column list, we do not expect value with dramatic fluctuation in neighbor weeks of the same show
    quality_metric['erroneous_data'] = 100
    col_list = ['FB Likes', 'FB Checkins', 'Twitter Followers', 'Instagram Followers']
    error = erroneous_data(df, col_list, status)
    counter = 0
    for col in col_list:
        quality_metric.loc[col,'erroneous_data'] = round(100- error.loc[counter,'error'],2)
        counter = counter + 1

    # Calculate total score, which it the mean of all the dimensions:
    counter=0
    for i in col_names[0:15]:
        quality_metric.loc[i,'total_score'] = quality_metric.iloc[counter,1:5].mean()
        counter = counter +1

    return(quality_metric.iloc[0:15,])


def missing_value(df):
    # count missing value in the dataframe
    missingValue = df.loc[df.isnull().sum(axis=1) > 0,].shape[0]
    # total numbers of value
    total_number = df.shape[0] * df.shape[1]
    # percentage of missing value:
    p_of_missing_value = missingValue/total_number
    return(p_of_missing_value)

def data_type(df,type_list):
    wrong_type = pd.DataFrame()
    for i in range(0,len(type_list)):
        wrong_type.loc[i, "wrong"] = 0
        for k in df.iloc[:,i]:
            if type(k) != type_list[i]:
                wrong_type.loc[i,"wrong"] = wrong_type.loc[i,"wrong"] +1
    # total number of values
    total_number = df.shape[0]
    # percentage of missing value:
    wrong_type = wrong_type / total_number
    return(wrong_type)

def redundancy(df):
    redundancy = 0
    for i in range(1,df.shape[0]):
        if df.loc[i,'Show'] ==df.loc[i-1,'Show'] and df.loc[i,'Date']== df.loc[i-1,'Date']:
            redundancy = redundancy +1
        i = i+1
    # total number of rows
    total_number = df.shape[0]
    # percentage of redundant rows:
    p_of_redundancy = redundancy / total_number
    return (p_of_redundancy)

def erroneous_data(df,col_list,status):
    erroneous = pd.DataFrame()
    counter = 0
    for col in col_list:
        erroneous.loc[counter,'error'] = 0
        if status == 'before':
            df[col] = df[col].str.replace(',','')
        for i in range(1,df[col].shape[0]-1):
            # In the same show, if a value in the column_list is far from the value in the weeks next to it,
            # it would be considered as erroneous data point
            if df.loc[i, 'Show'] == df.loc[i - 1, 'Show'] and df.loc[i, 'Show'] == df.loc[i+1, 'Show'] and \
                ((int(df.loc[i, col]) < 0.5 * int(df.loc[i - 1, col]) and int(df.loc[i, col]) < 0.5 * int(df.loc[i + 1, col])) or
                     (int(df.loc[i, col]) > 1.5 * int(df.loc[i - 1, col]) and int(df.loc[i, col]) > 1.5 * int(df.loc[i + 1, col]))):
                erroneous.loc[counter,'error']  =erroneous.loc[counter,'error'] + 1
        counter = counter + 1
    # total number of columns scanned
    #total_number = df.shape[0]
    # percentage of erroneous_data
    #p_of_erroneous_data = erroneous_data / total_number
    return(erroneous)


if __name__ == "__main__":
    main(sys.argv)