"""
This is the grosses data cleaning script
"""

import pandas as pd
import sys

pd.set_option('display.max_columns', 10) # set max num of cols displayed

# function 1(missing_clean): clean missing data
def missing_clean(df):

    # remove na data

    #takes a look at the na data, as well as rows before and after
    ind_na = df[df.show.isnull()].index[0]
    print(df.iloc[[ind_na-1, ind_na, ind_na+1],])

    # it is clear that the na data are a copy of the previous record, so we can simply remove it.
    df = df.drop(ind_na, axis = 0)

    # now remove zero data
    # zero grosses may be due to that fact that theathers are closed in holiday seasons
    # only consider shows that never have positive grosses in the entire data set to be missing

    total_zero_rows = df[(df.this_week_gross == '$0.00') & (df.diff_in_dollars == '$0.00')] # gets all zero rows

    shows_remove = []
    for show in total_zero_rows.show.unique():
        if df[(df.show == show) & (df.this_week_gross != "$0.00")].shape[0] == 0:
            shows_remove.append(show)

    for show in shows_remove:
        df = df[df["show"] != show]

    return (df)


# takes in a string
# removes dollar signs ($) and commas (,)
def remove_usd_comma(string):
    string = string.replace('$', '')
    commas_removed_string = string.replace(',', '')
    return(commas_removed_string)

# takes in a string
# remove percent signs (%)
def remove_percent(string):
    string = string.replace('%', '')
    return(string)

# function 2 (dataType_clean): clean $ and % and data types
def dataType_clean(df):

    # remove dollar and percent signs
    df["this_week_gross_2"] = df.apply(lambda x: remove_usd_comma(x["this_week_gross"]), axis = 1)
    df["diff_in_dollars_2"] = df.apply(lambda x: remove_usd_comma(x["diff_in_dollars"]), axis=1)
    df["avg_ticket_price_2"] = df.apply(lambda x: remove_usd_comma(x["avg_ticket_price"]), axis=1)

    df["percent_of_cap_2"] = df.apply(lambda x: remove_percent(x["percent_of_cap"]), axis=1)
    df["diff_percent_of_cap_2"] = df.apply(lambda x: remove_percent(x["diff_percent_of_cap"]), axis=1)

    # convert data types
    df['this_week_gross_2'] = df['this_week_gross_2'].astype(float)
    df['diff_in_dollars_2'] = df['diff_in_dollars_2'].astype(float)
    df['avg_ticket_price_2'] = df['avg_ticket_price_2'].astype(float)
    df['percent_of_cap_2'] = df['percent_of_cap_2'].astype(float)
    df['diff_percent_of_cap_2'] = df['diff_percent_of_cap_2'].astype(float)

    # convert percent into decimal numbers
    df['percent_of_cap_2'] = df['percent_of_cap_2'].div(100).round(4)
    df['diff_percent_of_cap_2'] = df['diff_percent_of_cap_2'].div(100).round(4)

    # drop old columns
    df = df.drop(["this_week_gross", "diff_in_dollars", "avg_ticket_price", "percent_of_cap", "diff_percent_of_cap"], axis = 1)

    # rename columns
    names = ['week_ending', 'show', 'seats_sold', 'perfs', 'this_week_gross',
       'diff_in_dollars', 'avg_ticket_price', 'percent_of_cap',
       'diff_percent_of_cap']
    df.columns = names

    # reorder columns
    df = df.reindex(columns=['week_ending', 'show', 'this_week_gross',
       'diff_in_dollars', 'avg_ticket_price', 'seats_sold', 'perfs',  'percent_of_cap',
       'diff_percent_of_cap'])

    return(df)


# function 3: clean redundancy; keep first occurrences

def redundancy_clean(df):
    df = df.drop_duplicates(['show', 'this_week_gross', 'diff_in_dollars'])
    df = df[df.week_ending != "1985-06-02"]

    return (df)

def main(argv):

    # read in the data set
    df = pd.read_csv("broadway_grosses.csv", encoding = "latin1")

    # performs data cleaning
    df = missing_clean(df)
    df= dataType_clean(df)
    df = redundancy_clean(df)

    df = df.sort_values(by="week_ending", ascending= False)

    df.to_csv("grosses_cleaned.csv", index = False)

if __name__ == "__main__":
    main(sys.argv)