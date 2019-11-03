import pandas as pd
import sys
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


def reassign_missing_w_mean(df,col_name):
    new_col_name = 'cleaned_'+col_name
    for i in range(0,df.shape[0]):
        if df.loc[i,col_name] <=0  and df.loc[i,'Current']=='Current':
            show_name = df.loc[i,'Show']
            df.loc[i, new_col_name] = df.loc[(df['Show']==show_name) & (df[col_name]>0),new_col_name].mean()
        else:
            df.loc[i, new_col_name] =df.loc[i, col_name]
    return(df)

def vs_last_week(df,col_name,vs_col_name):
    for i in range(0,df.shape[0]-1):
        if df.loc[i, 'Show'] == df.loc[i + 1, 'Show']:
            df.loc[i, vs_col_name] = df.loc[i, col_name]- df.loc[i+1, col_name]
        else:
            df.loc[i, vs_col_name] =0
    return(df)

def lof(df):
    outliers = []
    clf=LocalOutlierFactor(n_neighbors=5,algorithm='auto',contamination=0.1,n_jobs=-1,p=2)
    list = df['Show'].unique()
    for show in list:
        data= df.loc[df['Show']==show].iloc[:,2:12]
        data = data.fillna(0)
        if len(data)>=5:
            outlier = clf.fit_predict(data)
            outliers=np.append(outliers,outlier)
        else:
            for i in range(0,len(data)):
                outliers = np.append(outliers,0)
    df['outlier_detection'] = outliers
    return(df)

def description(df,col_list):
    df1 = df[col_list]
    df_mean = pd.DataFrame()
    df_mean['mean'] = df1.mean()
    df_mean['median'] = df1.median()
    df_mean['sd'] = df1.std()
    print(df_mean)


def main(argv):
    ## read in the data set
    df = pd.read_csv('cleaned_SocialMedia.csv', sep=',', encoding='latin1')

    ## sort the data frame by Show Name, descending the date
    df = df.sort_values(by=['Show', 'Year', 'Month', 'Day'], ascending=(True, False, False, False))
    df = df.reset_index(drop=True)

    ## if there was any missing value, we used the mean of the show to reassign it
    ## reassignment only applied on current shows (not upcomming ones), because the upcomming shows might have '0' talking about which should not be treated as missing value
    df = reassign_missing_w_mean(df, 'FB Talking About')
    df = reassign_missing_w_mean(df, 'FB Checkins')

    ## reorder the dataframe after cleaning
    order = ['Show', 'Date', 'FB Likes', 'Likes Vs.Last Week', 'cleaned_FB Talking About', 'Talking Vs.Last Week',
             'cleaned_FB Checkins', 'Checkins vs.Last Week', 'Twitter Followers', 'Twitter vs.Last Week',
             'Instagram Followers', 'IG Followers vs.Last Week', 'Current', 'Type', 'FB Talking About', 'FB Checkins',
             'Month', 'Day', 'Year']
    df = df[order]

    ## recalculate the columns of comparing this week v.s. last week
    df = vs_last_week(df, 'FB Likes', 'Likes Vs.Last Week')
    df = vs_last_week(df, 'cleaned_FB Talking About', 'Talking Vs.Last Week')
    df = vs_last_week(df, 'cleaned_FB Checkins', 'Checkins vs.Last Week')
    df = vs_last_week(df, 'Twitter Followers', 'Twitter vs.Last Week')
    df = vs_last_week(df, 'Instagram Followers', 'IG Followers vs.Last Week')

    ## detect outliers -- might be considered as extra credit :)
    df = lof(df)

    ## output the mean, median and standard deviation
    col_list = ['Likes Vs.Last Week', 'cleaned_FB Talking About', 'Checkins vs.Last Week', 'Twitter vs.Last Week',
                'IG Followers vs.Last Week']
    description(df, col_list)

    ## save the file
    df.to_csv('part2cleanedSocialMedia.csv')


if __name__ == "__main__":
    main(sys.argv)