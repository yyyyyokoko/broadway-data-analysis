# Loading Libraries
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pprint

######################################################
# Clustering Analysis
######################################################


def clusteringAnalysis(df, method):

    """
    :param df: read into the data frame of interest - part2cleanedSocialMedia.csv
    :param method: clustering methods used in this analysis: Ward, K-Means, DBSCAN
    :return: the quality scores of the clusters for the three clustering methods as well as scatter plots
    """

    # convert categorical values to numerical values for kmeans
    df['Current'] = pd.Categorical(df['Current'])
    df['Current'] = df['Current'].cat.codes

    df = pd.concat([df['Likes Vs.Last Week'], df['Twitter vs.Last Week'], df['IG Followers vs.Last Week']], axis=1,
                   keys=['Likes Vs.Last Week', 'Twitter vs.Last Week', 'IG Followers vs.Last Week'])

    x = df.values          # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    # clustering = method.fit(normalizedDataFrame)
    cluster_labels = method.fit_predict(normalizedDataFrame)

    # Silhouette Procedures
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    silhouette = "The average silhouette_score is : " + str(silhouette_avg)

    # Calinski-Harabaz Procedures (https: // scikit - learn.org / stable / modules / generated /
    # sklearn.metrics.calinski_harabasz_score.html  # sklearn.metrics.calinski_harabasz_score
    calinski_harabaz = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    calinski = "The average calinski_harabaz score is : " + str(calinski_harabaz)

    output = "The quality of the clusters using the clustering method " + str(method) + " has the following scores: " \
             + "\n" + silhouette + '\n' + calinski

    # Plot the clusters
    plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=cluster_labels, cmap="plasma")
    # plt.show()

    return output


#####################################################
# Association Rules / Frequent Itemset Mining Analysis
#####################################################


def associationRule(df):

    """
    :param df: Input the Gross dataframe - part2cleanedGrosses.csv
    :return: Print out and save the itemsets for three different support values and calculate the confidence.
    """

    # Generating the itemsets
    df['week_ending'] = pd.to_datetime(df['week_ending']).dt.strftime('%Y-%m-%d')
    gross_date = [x for x in gross['week_ending'].unique() if int(x[0:4]) > 2000]
    all_date = df['week_ending'].unique()

    itemset = []
    for i in all_date:
        temp = list(gross.loc[(df['week_ending'] == i) & df['percent_of_cap'] >= 0.8, 'show'])
        itemset.append(temp)

    itemset2 = []
    for i in gross_date:
        temp = list(gross.loc[(df['week_ending'] == i) & df['percent_of_cap'] >= 0.8, 'show'])
        itemset2.append(temp)

    # Perform Apriori algorithm
    for j in range(0, 2):
        temp = [itemset, itemset2][j]
        te = TransactionEncoder()
        te_ary = te.fit(temp).transform(temp)
        temp_df = pd.DataFrame(te_ary, columns=te.columns_)
        value = [0.4, 0.6, 0.8]  # support value

        confidf = pd.DataFrame()
        supportdf = pd.DataFrame()
        for i in value:
            frequent_itemsets =  apriori(temp_df, min_support=i,use_colnames=True)
            confi = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
            frequent_itemsets['support_val'] = i
            confi['support_val'] = i
            supportdf = supportdf.append(frequent_itemsets)
            confidf = confidf.append(confi.iloc[:,[0,1,5,-1]])
            print('#################### Support =', i, '####################')
            pprint.pprint(frequent_itemsets)
            print('####### Calculating the Confidence #######')
            print(confi.iloc[:,[0,1,5,-1]])
        supportdf.to_csv(str('Itemset_support' + str(j) + '.csv'), index=False)
        confidf.to_csv(str('Itemset_confidence' + str(j) + '.csv'), index=False)


if __name__ == "__main__":

    ######################################################
    # Clustering Analysis
    ######################################################
    # Read in data directly into pandas
    myData = pd.read_csv('part2cleanedSocialMedia.csv', sep=',', encoding='latin1')

    # Add each CLUSTER method and its name to the model array
    methods = [('Ward', AgglomerativeClustering()), ('K-Means', KMeans(n_clusters=3)),
               ('DBSCAN', DBSCAN(eps=0.1, min_samples=2))]

    # Write to file: the quality scores of the clusters using the Silhouette & Calinski-Harabaz procedures
    with open("clustering-quality.txt", 'w', encoding='utf-8') as f:
        for method in methods:
            file = f.writelines(clusteringAnalysis(myData, method[1]) + '\n\n')

    ######################################################
    # Association Rules / Frequent Itemset Mining Analysis
    ######################################################
    gross = pd.read_csv("part2cleanedGrosses.csv", index_col=0, encoding='latin-1')
    associationRule(gross)
