#import packages
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pprint

def associationRule(df):
    #Input the Gross dataframe
    #Print out and save the itemsets for three different support values and calculate the confidence.

    #Generating the itemsets
    df['week_ending'] = pd.to_datetime(df['week_ending']).dt.strftime('%Y-%m-%d')
    gross_date = [x for x in gross['week_ending'].unique() if int(x[0:4]) > 2000]
    all_date = df['week_ending'].unique()

    itemset = []
    for i in all_date:
        temp = list(gross.loc[(df['week_ending']== i) & df['percent_of_cap'] >= 0.8, 'show'])
        itemset.append(temp)

    itemset2 = []
    for i in gross_date:
        temp = list(gross.loc[(df['week_ending']== i) & df['percent_of_cap'] >= 0.8, 'show'])
        itemset2.append(temp)

    #Perform Apriori algorithm
    for j in range(0, 2):
        temp = [itemset, itemset2][j]
        te = TransactionEncoder()
        te_ary = te.fit(temp).transform(temp)
        temp_df = pd.DataFrame(te_ary, columns=te.columns_)
        value = [0.4, 0.6, 0.8] #support value

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
        confidf.to_csv(str('Itemset_confidence'+ str(j) + '.csv'), index=False)

if __name__ == "__main__":
    gross = pd.read_csv("part2cleanedGrosses.csv", index_col=0, encoding='latin-1')
    associationRule(gross) 