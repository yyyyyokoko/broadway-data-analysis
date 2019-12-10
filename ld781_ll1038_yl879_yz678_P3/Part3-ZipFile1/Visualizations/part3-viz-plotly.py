import sys
import string
import re
import chart_studio
import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

######################################################
# Data Preparation for Visualizations 1 & 2
######################################################


def clean_data(gross, social):
    """
    In this function, we modify the datatype for further visualization tasks
    :param gross: the Broadway Grosses Data Set
    :param social: the Broadway Social Stats Data Set
    :return: the cleaned data sets
    """

    # modify data type
    social = social.drop(columns=['Month', 'Day', 'Year', 'outlier_detection'])
    social['Date'] = pd.to_datetime(social['Date']).dt.strftime('%Y-%m-%d')
    gross['week_ending'] = pd.to_datetime(gross['week_ending']).dt.strftime('%Y-%m-%d')
    social['Show'] = [re.sub(r'\s+', ' ', x.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower().strip()) for x in social['Show']]
    # ain't too proud has weird special characters
    gross['show'] = [x.split('¡')[0] for x in gross['show']]
    # beautiful carol king has different names
    gross['show'] = [re.sub(r'\s+', ' ', x.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower().strip()) for x in gross['show']]
    gross.loc[gross['show'] == 'beautiful the carole king musical', 'show'] = 'beautiful'

    return gross, social


def preprocessing(gross, social):
    """
    In this function, we merge two datasets together for further visualization tasks
    :param gross: the cleaned Broadway Grosses Data Set
    :param social: the cleaned Broadway Social Stats Data Set
    :return: the merged data set
    """

    # Match the dates from two data sets
    gross_date = [x for x in gross['week_ending'].unique() if x[0:4] in ['2019', '2018', '2017']]
    temp_date = [x.date().strftime('%Y-%m-%d') for x in list(pd.to_datetime(gross_date) + pd.Timedelta(1, unit='d'))]
    gross_date = gross_date + temp_date

    names = list(social.columns)
    names.append('this_week_gross')
    df = pd.DataFrame(columns=names)

    for i in social['Date'].unique():
        temp = pd.to_datetime(i).date().strftime('%Y-%m-%d')
        if temp in gross_date:
            temp_df = social.loc[social['Date'] == temp, :]
            c = gross.loc[(gross['week_ending'] == temp), ['week_ending', 'show', 'this_week_gross']]
            for j in temp_df.Show:
                for k in c.show:
                    if j in k:
                        temp_df.loc[(temp_df['Show'] == j), 'this_week_gross'] = c.loc[
                            c['show'] == k, 'this_week_gross'].values
                    elif k in j:
                        temp_df.loc[(temp_df['Show'] == j), 'this_week_gross'] = c.loc[
                            c['show'] == k, 'this_week_gross'].values
            df = df.append(temp_df, ignore_index=True)
    df_notnull = df.dropna()
    return df_notnull


######################################################
# Visualization 1: Plotly & Interactive
######################################################


def followers(df_notnull):
    """
    In this function, we create the visualization task for follower comparasions
    :param df_notnull: merged dataset
    :return: showing the visualization in a webpage
    """
    t = df_notnull.loc[df_notnull['Show'] == 'aladdin', ['Date', 'Twitter Followers', 'Instagram Followers']]
    data = []
    for i in df_notnull.Show.unique():
        a = df_notnull.loc[df_notnull['Show'] == i, ['Date', 'Twitter Followers', 'Instagram Followers']]
        x = a.Date.values
        data.append(go.Scatter(x=x, y=a['Twitter Followers'].values, name=str(i) + ' Twitter Followers'))
        data.append(go.Scatter(x=x, y=a['Instagram Followers'].values, name=str(i) + ' IG Followers'))

    layout = go.Layout(barmode='stack', updatemenus=[go.layout.Updatemenu(
        buttons=list([
            dict(label="Ain't too proud",
                 method="update",
                 args=[{"visible": [True] * 2 + [False] * 52},
                       {"title": "Follower Variation of Ain't too proud"}]),
            dict(label="Aladdin",
                 method="update",
                 args=[{"visible": [False] * 2 + [True] * 2 + [False] * 50},
                       {"title": "Follower Variation of Aladdin"}]),
            dict(label="Beautiful: The Carole King",
                 method="update",
                 args=[{"visible": [False] * 4 + [True] * 2 + [False] * 48},
                       {"title": "Follower Variation of Beautiful: The Carole King"}]),
            dict(label="Beetlejuice",
                 method="update",
                 args=[{"visible": [False] * 6 + [True] * 2 + [False] * 46},
                       {"title": "Follower Variation of Beetlejuice"}]),
            dict(label="Betrayal",
                 method="update",
                 args=[{"visible": [False] * 8 + [True] * 2 + [False] * 44},
                       {"title": "Follower Variation of Betrayal"}]),
            dict(label="Book of Mormon",
                 method="update",
                 args=[{"visible": [False] * 10 + [True] * 2 + [False] * 42},
                       {"title": "Follower Variation of Book of Mormon"}]),
            dict(label="Chicago",
                 method="update",
                 args=[{"visible": [False] * 12 + [True] * 2 + [False] * 40},
                       {"title": "Follower Variation of Chicago"}]),
            dict(label="Come from away",
                 method="update",
                 args=[{"visible": [False] * 14 + [True] * 2 + [False] * 38},
                       {"title": "Follower Variation of Come from away"}]),
            dict(label="Dear Evan Hansen",
                 method="update",
                 args=[{"visible": [False] * 16 + [True] * 2 + [False] * 36},
                       {"title": "Follower Variation of Dear Evan Hansen"}]),
            dict(label="Freestyle Love Supreme",
                 method="update",
                 args=[{"visible": [False] * 18 + [True] * 2 + [False] * 34},
                       {"title": "Follower Variation of Freestyle Love Supreme"}]),
            dict(label="Frozen",
                 method="update",
                 args=[{"visible": [False] * 20 + [True] * 2 + [False] * 32},
                       {"title": "Follower Variation of Frozen"}]),
            dict(label="Hadestown",
                 method="update",
                 args=[{"visible": [False] * 22 + [True] * 2 + [False] * 30},
                       {"title": "Follower Variation of Hadestown"}]),
            dict(label="Hamilton",
                 method="update",
                 args=[{"visible": [False] * 24 + [True] * 2 + [False] * 28},
                       {"title": "Follower Variation of Hamilton"}]),
            dict(label="Mean Girls",
                 method="update",
                 args=[{"visible": [False] * 26 + [True] * 2 + [False] * 26},
                       {"title": "Follower Variation of Mean Girls"}]),
            dict(label="Moulin Rouge!",
                 method="update",
                 args=[{"visible": [False] * 28 + [True] * 2 + [False] * 24},
                       {"title": "Follower Variation of Moulin Rouge!"}]),
            dict(label="Oklahoma!",
                 method="update",
                 args=[{"visible": [False] * 30 + [True] * 2 + [False] * 22},
                       {"title": "Follower Variation of Oklahoma!"}]),
            dict(label="Sea Wall/A Life",
                 method="update",
                 args=[{"visible": [False] * 32 + [True] * 2 + [False] * 20},
                       {"title": "Follower Variation of Sea Wall/A Life"}]),
            dict(label="Slave Play",
                 method="update",
                 args=[{"visible": [False] * 34 + [True] * 2 + [False] * 18},
                       {"title": "Follower Variation of Slave Play"}]),
            dict(label="The Great Society",
                 method="update",
                 args=[{"visible": [False] * 36 + [True] * 2 + [False] * 16},
                       {"title": "Follower Variation of The Great Society"}]),
            dict(label="The Lightning Thief",
                 method="update",
                 args=[{"visible": [False] * 38 + [True] * 2 + [False] * 14},
                       {"title": "Follower Variation of The Lightning Thief"}]),

            dict(label="The Lion King",
                 method="update",
                 args=[{"visible": [False] * 40 + [True] * 2 + [False] * 12},
                       {"title": "Follower Variation of The Lion King"}]),
            dict(label="The Phantom of the Opera",
                 method="update",
                 args=[{"visible": [False] * 42 + [True] * 2 + [False] * 10},
                       {"title": "Follower Variation of The Phantom of the Opera"}]),
            dict(label="The Sound Inside",
                 method="update",
                 args=[{"visible": [False] * 44 + [True] * 2 + [False] * 8},
                       {"title": "Follower Variation of The Sound Inside"}]),
            dict(label="Tootsie",
                 method="update",
                 args=[{"visible": [False] * 46 + [True] * 2 + [False] * 6},
                       {"title": "Follower Variation of Tootsie"}]),
            dict(label="Waitress",
                 method="update",
                 args=[{"visible": [False] * 48 + [True] * 2 + [False] * 4},
                       {"title": "Follower Variation of Waitress"}]),
            dict(label="Wicked",
                 method="update",
                 args=[{"visible": [False] * 50 + [True] * 2 + [False] * 2},
                       {"title": "Follower Variation of Wicked"}]),
            dict(label="Harry Potter and the Cursed Child",
                 method="update",
                 args=[{"visible": [False] * 52 + [True] * 2},
                       {"title": "Follower Variation of Harry Potter and the Cursed Child "}]),
        ]))],
                       title=go.layout.Title(text='Follower Variation of Broadway Shows'),
                       xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Date')),
                       yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Followers')))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='Followers', auto_open=True)
    return


######################################################
# Visualization 2: Plotly & Interactive
######################################################


def FBvsGross(df_notnull):
    """
    In this function, we create the visualization task for the ratio of FB likes and Gross
    :param df_notnull: merged dataset
    :return: showing the visualization in a webpage
    """
    new = df_notnull.loc[df_notnull.Date == '2019-09-22', ['Show', 'FB Likes', 'this_week_gross']]
    shows = new.Show.unique()
    for i in shows:
        temp = df_notnull.loc[df_notnull.Show == i, 'this_week_gross'].mean()
        new.loc[(new['Show'] == i), 'this_week_gross'] = temp
    new['size'] = 3
    fig = px.scatter(new, x="FB Likes", y="this_week_gross", color="Show", size='size',
                     title="FB likes vs. average weekly gross")

    py.plot(fig, filename='FBlikes', auto_open=True)
    return


######################################################
# Visualization 3: Plotly & Interactive
######################################################


def viz(x, y1, y2, filename, title1, title2, title3):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1,
                             mode='lines',
                             name=title1))
    fig.add_trace(go.Scatter(x=x, y=y2,
                             mode='lines+markers',
                             name=title2))
    fig.update_layout(title=title3)
    py.plot(fig, filename=filename, auto_open=True)
    fig.show()


######################################################
# Visualization 4 & 5: Plotly & Interactive
######################################################

def main(argv):
    # plot rating against gross
    draw_gross_rating()
    # plot price against percent of cap
    draw_cap_price()

######################################################


def draw_gross_rating():
    """
    The function plots total rating against average weekly gross
    """

    # read in data
    gs = pd.read_csv("part2cleanedGrosses.csv", sep=',', encoding='latin1')
    rt = pd.read_csv("Musical_ratings-withoutNa-cleaned.csv")

    # prepare data

    # limit the time scope to recent 5 years
    testData2 = gs[gs['year'] >= 2015]

    testData2 = testData2[['show', 'year', 'month', 'this_week_gross']]

    # calculate avg weekly grosses mean (by show)
    testData2['avg_weekly_gross'] = testData2.groupby('show')['this_week_gross'].transform('mean')
    testData2_1 = pd.merge(testData2, rt, on='show')
    # select distinct show
    testData2_1 = testData2_1.drop_duplicates('show')

    # Select relevant columns
    testData2_1 = testData2_1[['show', 'avg_weekly_gross', 'total_rating']]

    fig = px.scatter(testData2_1, x="avg_weekly_gross", y="total_rating", color="total_rating", hover_name="show",
                     labels=dict(avg_weekly_gross="Average Weekly Grosses (in USD)", total_rating="Ratings"),
                     color_continuous_scale=px.colors.colorbrewer.RdYlGn,
                     title="Ratings vs. Average weekly grosses of recent 5 years")
    py.plot(fig, "gross-rating", auto_open=True)
    # save a local graph
    # pio.write_html(fig, file='grosses_rating.html', auto_open=True)


##################################################

def draw_cap_price():
    """
    the function plot the average ticket price against percent of cap for the recent 5 years
    """

    # read in data file
    gs = pd.read_csv("part2cleanedGrosses.csv", sep=',', encoding='latin1')

    # prepare the data
    gs = gs[gs['year'] > 2015]
    data = gs[(gs['avg_ticket_price'] > 0) & (gs['percent_of_cap'] > 0)]
    data = data[['week_ending', 'show', 'this_week_gross', 'avg_ticket_price', 'percent_of_cap']]

    # plot the data
    fig = px.scatter(data, x="percent_of_cap", y='avg_ticket_price', color='show', hover_name="show", width=1400,
                     height=400, labels=dict(avg_ticket_price="Average Ticket Price (in USD)",
                                             percent_of_cap="Percent of Cap"),
                     title="  Average Ticket Price vs. Percent of Cap  of recent 5 years")
    py.plot(fig, "price-cap", auto_open=True)


if __name__ == "__main__":

    ######################################################
    # Visualizations 1 & 2: Credentials from Luwei
    ######################################################
    chart_studio.tools.set_credentials_file(username='yyyyyokoko', api_key='NoVGkxcKvi17mgozh7kJ', )
    chart_studio.tools.set_config_file(world_readable=True, sharing='public')
    social = pd.read_csv("part2cleanedSocialMedia.csv", index_col=0, encoding='latin-1')
    gross = pd.read_csv("grosses_cleaned.csv")
    gross, social = clean_data(gross, social)
    df_cleaned = preprocessing(gross, social)
    followers(df_cleaned)  # https://plot.ly/~yyyyyokoko/25/#/
    FBvsGross(df_cleaned)  # https://plot.ly/~yyyyyokoko/32/#/

    ######################################################
    # Visualization 3: Credentials from Janet
    ######################################################
    chart_studio.tools.set_credentials_file(username='janetlauyeung', api_key='RKYvimfhbva2IYWSQQAx')
    df1 = pd.read_csv('Musical_ratings-withoutNa-cleaned.csv')
    viz(df1["show"], df1["critics_rating"], df1["readers_rating"], "viz-ratings", "critics_rating", "readers_rating",
        "Critics' and Readers' Ratings for Broadway Shows")  # https://chart-studio.plot.ly/~janetlauyeung/76

    ######################################################
    # Visualizations 4 & 5: Credentials from Lujia
    ######################################################
    chart_studio.tools.set_credentials_file(username='KaluluD', api_key='HPujkTbNCxTcfhYkRcbP')
    chart_studio.tools.set_config_file(world_readable=True, sharing='public')
    main(sys.argv)
