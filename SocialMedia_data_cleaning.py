import pandas as pd
import sys

def remove_na(df):
    df = df.dropna()
    return(df)

# change the data type
def change_data_type(df):
    for col in list(df):
        df[col] = df[col].str.replace(',', '')
        df[col] = df[col].str.replace('%', '')
    df.iloc[:,2:12] = df.iloc[:,2:12].astype(int)
    df['Total Fans Change'] = df['Total Fans Change'].astype(float)
    df['Total Fans Change'] =df['Total Fans Change'] /100
    return (df)


def year_clean(df):
    df_date = df['Date'].str.split('/',expand=True)
    df['Month']=df_date[0]
    df['Day'] = df_date[1]
    df.loc[0,'Year'] = 2019
    df.loc[0,'Date'] = str(int(df.loc[0,'Year'])) +'/'+df.loc[0,'Month']+'/'+df.loc[0,'Day']
    for i in range(1,len(df['Date'])):
        if df.loc[i,'Show'] ==df.loc[i-1,'Show']:
            if int(df.loc[i,'Month'])>int(df.loc[i-1,'Month']):
                df.loc[i,'Year'] = df.loc[i-1,'Year']-1
            else:
                df.loc[i, 'Year'] = df.loc[i-1, 'Year']
        else:
            df.loc[i, 'Year'] = 2019
        df.loc[i,'Date'] = str(int(df.loc[i,'Year'])) +'/'+df.loc[i,'Month']+'/'+df.loc[i,'Day']
    df.to_csv('cleaned_SocialMedia.csv',index = False)
    return(df)

def remove_redundancy(df):
    list = []
    for i in range(1,df.shape[0]):
        if df.loc[i,'Show'] ==df.loc[i-1,'Show'] and df.loc[i,'Day']== df.loc[i-1,'Day'] and df.loc[i,'Month']== df.loc[i-1,'Month']:
            list.append(i)
    df = df.drop(list, axis=0)
    df = df.reset_index(drop=True)
    return(df)

def correct_erroneous(df,col_list):
    for col in col_list:
        for i in range(1, df[col].shape[0] - 1):
            # In the same show, if a value in the column_list is far from the value in the weeks next to it,
            # it would be considered as erroneous data point
            # to clean it, assign it an average value of weeks next to it
            if df.loc[i, 'Show'] == df.loc[i - 1, 'Show'] and df.loc[i, 'Show'] == df.loc[i + 1, 'Show'] and \
                    ((int(df.loc[i, col]) < 0.5 * int(df.loc[i - 1, col]) and int(df.loc[i, col]) < 0.5 * int(
                        df.loc[i + 1, col])) or
                     (int(df.loc[i, col]) > 1.5 * int(df.loc[i - 1, col]) and int(df.loc[i, col]) > 1.5 * int(
                         df.loc[i + 1, col]))):
                df.loc[i, col] = (df.loc[i-1, col] + df.loc[i+1, col])/2
    df = df.reset_index(drop=True)
    return(df)

def main(argv):
    df = pd.read_csv('scrap_of_social_media.csv' , sep=',', encoding='latin1')
    df1 = remove_na(df)
    df2 = change_data_type(df1)
    df3 = year_clean(df2)
    col_list = ['FB Likes', 'FB Checkins', 'Twitter Followers', 'Instagram Followers']
    df4 = correct_erroneous(df3, col_list)
    df5 = remove_redundancy(df4)
    df5.to_csv('cleaned_SocialMedia.csv',index = False)

if __name__ == "__main__":
    main(sys.argv)