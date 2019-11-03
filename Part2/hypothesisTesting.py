import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.formula.api as smf
import sys

def main(argv):
    # read in the data set
    df = pd.read_csv('part2cleanedGrosses.csv' , sep=',', encoding='latin1')
    df = df.drop('Unnamed: 0',axis= 1)

    gs = pd.read_csv("part2cleanedGrosses.csv", sep=',', encoding='latin1')
    rt = pd.read_csv("Musical_ratings-withoutNa-cleaned.csv")

    # Hypothesis 1

    # split the date in preparation of further analysis
    print("Splitting dates...")
    df = split_date(df)
    print()
    # categorize records into time periods: Thanksgiving, summer break, winter break and not holiday
    print("Setting time periods...")
    df = set_time_period(df)
    print()
    # ANOVA analysis about whether the seats sold mean among different groups are significantly different
    print("Testing the first hypothesis...")
    print()
    anova_analysis(df)
    print()

    # Hypothesis 2
    print("Testing the second hypothesis...")
    print()
    hypoTest2(gs, rt)
    print()
    # Hypothesis 3
    print("Testing the third hypothesis...")
    print()
    hypoTest3(gs)
    print()

##################################################

def split_date(df):
    """
    split_date takes in a dataframe
    splits the year, month, and day for further analysis
    and returns a modified data frame
    """
    df_date = df['week_ending'].str.split('-', expand=True)
    df['Year'] = df_date[0]
    df['Month'] = df_date[1]
    df['Day'] = df_date[2]

    # sort data set by date
    df = df.sort_values(by = ['Year','Month','Day'],ascending=(False,False,False))

    # convert date type
    df['Year']=df['Year'].astype(int)
    df['Month']=df['Month'].astype(int)
    df['Day']=df['Day'].astype(int)

    # convert date to timestamp
    #df['week_ending']=df['week_ending'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
    df['week_ending']=pd.to_datetime(df['week_ending'])
    return(df)

##################################################

def set_time_period(df):
    """
    set_time_period takes in a data frame
    and calculates the time period (ThanksGiving, WinterBreak, SummerBreak or Not Holiday)for each record
    returns a modified data frame
    """
    for i in range(0,len(df['Day'])):
        if (((df.loc[i,'Month']==11) & (df.loc[i,'Day']<=30) & (df.loc[i,'Day']>=27))\
            | ((df.loc[i,'Month']==12) & (df.loc[i,'Day']<=3) & (df.loc[i,'Day']>=1)) ):
            df.loc[i,'Holiday'] = 'ThanksGiving'
            TGDate = df.loc[i,'week_ending']+timedelta(days=-7)
            df.loc[df['week_ending'] == TGDate,'Holiday'] = 'ThanksGiving'
        elif ((df.loc[i,'Month']==1) & (df.loc[i,'Day']<=11) & (df.loc[i,'Day']>=5)):
            df.loc[i,'Holiday'] = 'WinterBreak'
            WBDate = df.loc[i,'week_ending']+timedelta(days=-7)
            df.loc[df['week_ending'] == WBDate, 'Holiday'] = 'WinterBreak'
        elif ((df.loc[i,'Month']==7) | (df.loc[i,'Month']==8)):
            df.loc[i, 'Holiday'] = 'SummerBreak'
        else:
            df.loc[i, 'Holiday'] = 'Not Holiday'
    return(df)

##################################################

def anova_analysis(df):
    """
    anova_analysis takes in a data frame and performs an anova test for hypothesis testing 1
    prints out the test results
    """
    time_periods = df.groupby(['week_ending','Holiday'],as_index = False)[['seats_sold']].sum()
    TG = time_periods.loc[time_periods['Holiday'] == 'ThanksGiving','seats_sold']
    WB = time_periods.loc[time_periods['Holiday'] == 'WinterBreak','seats_sold']
    SB = time_periods.loc[time_periods['Holiday'] == 'SummerBreak','seats_sold']
    NH = time_periods.loc[time_periods['Holiday'] == 'Not Holiday','seats_sold']
    f,p = stats.f_oneway(TG,WB,SB,NH)
    print('The f and p of ANOVA analysis are:')
    print(f,p)

    ## plot the mean of each group
    time_periods.boxplot('seats_sold', by='Holiday', figsize=(12, 8))
    fileName = 'ANOVA.png'
    plt.savefig(fileName)

    print("The mean seats sold of each time periods:")
    print(time_periods.groupby('Holiday')['seats_sold'].mean())

    pairwise = MultiComparison(time_periods['seats_sold'], time_periods['Holiday'])
    result = pairwise.tukeyhsd()
    print(pairwise)
    print(result)
    #print(pairwise.groupsunique)

##################################################

def hypoTest2(df, rt):
    """
    test2 takes in the grosses data set and the rating data set
    prepares the data
    performs a logistic regression to test the hypothesis 2
    prints out the regression results
    """
    from sklearn import preprocessing
    import statsmodels.api as sm

    gs = df
    ratings = rt

    # limit the time scope to recent 5 years
    testData2 = gs[gs['year']>=2015]

    testData2 = testData2[['show','year','month','this_week_gross']]

    # calculate avg weekly grosses mean (by show)
    testData2['avg_weekly_gross'] = testData2.groupby('show')['this_week_gross'].transform('mean')
    testData2_1 = pd.merge(testData2, ratings, on='show')
    # select distinct show
    testData2_1 = testData2_1.drop_duplicates('show')

    # Select relevant columns
    testData2_1 = testData2_1[['show', 'avg_weekly_gross', 'total_rating']]

    testData2_1['ratingLevel'] = 0
    testData2_1.loc[(testData2_1.total_rating > 7), 'ratingLevel'] = 1

    # normalize avg_weekly_gross
    mm_scaler = preprocessing.MinMaxScaler()
    mm_scaler.fit(testData2_1[['avg_weekly_gross']])
    testData2_1['norm_gross'] = mm_scaler.transform(testData2_1[['avg_weekly_gross']])

    # logistic regression
    X = sm.add_constant(testData2_1['norm_gross'])

    logit1 = sm.Logit(testData2_1['ratingLevel'], X)

    result1 = logit1.fit()

    # summarize the results
    print(result1.summary())

    # get the odds
    print()
    print("The odds-ratios are as the following:")
    print()
    print(np.exp(result1.params))

##################################################

def hypoTest3(df):
    """
    test3 takes in the grosses data set
    prepares the data
    performs a linear regression to test the hypothesis 3
    prints out the regression results
    """
    gs = df

    # Calculate Theatre size
    # account for cases with 0s

    gs['num_of_seats'] = 0
    gs.loc[(gs.seats_sold > 0) & (gs.perfs > 0) & (gs.percent_of_cap > 0), 'num_of_seats'] = (gs.seats_sold/gs.percent_of_cap/gs.perfs)

    testData = gs[(gs['num_of_seats'] > 0) & (gs['avg_ticket_price'] > 0)]
    testData = testData[['show','avg_ticket_price', 'num_of_seats', 'month', 'year']]

    # delete duplicates
    testData = testData.sort_values('num_of_seats').drop_duplicates('show')
    testData.hist()

    # normalize testData
    testData['log_num_seats'] = np.log(testData[['num_of_seats']])
    testData['log_price'] = np.log(testData[['avg_ticket_price']])

    lm_results = smf.ols('log_price ~ log_num_seats + month + year', data = testData).fit()

    print(lm_results.summary())

##################################################

if __name__ == "__main__":
    main(sys.argv)