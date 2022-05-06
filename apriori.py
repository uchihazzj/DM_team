from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np


if __name__ == '__main__':
    ratings_path = "data_clean/ratings.csv"
    ratings_df = pd.read_csv(ratings_path, encoding="utf_8_sig")
    ratings_df.drop(['RATING_ID','RATING_TIME'], axis=1, inplace=True)
    print(ratings_df.duplicated(['USER_MD5', 'MOVIE_ID']).sum())
    print(len(ratings_df['MOVIE_ID'].unique()) )
    ratings_count = ratings_df.groupby(['MOVIE_ID'])['RATING'].count().reset_index()
    ratings_count = ratings_count.rename(columns={'RATING':'totalRatings'}) 
    print(ratings_count.head(5))
    ratings_total = pd.merge(ratings_df,ratings_count, on='MOVIE_ID', how='left') 
    print(ratings_total.head(5))
    print(ratings_count['totalRatings'].describe())
    ratings_count.hist()
    print(ratings_count['totalRatings'].quantile(np.arange(.6,1,0.01)))
    votes_count_threshold = 320 #前2%
    ratings_top = ratings_total.query('totalRatings > @votes_count_threshold') 
    print(ratings_top.head(10))
    print(ratings_top.isna().sum())
    print(ratings_top.duplicated(['USER_MD5','MOVIE_ID']).sum())
    ratings_top = ratings_top.drop_duplicates(['USER_MD5','MOVIE_ID']) 
    print(ratings_top.duplicated(['USER_MD5','MOVIE_ID']).sum())
    df_for_apriori = ratings_top.pivot(index='USER_MD5',columns='MOVIE_ID',values='RATING') 
    print(df_for_apriori.head(5))
    df_for_apriori = df_for_apriori.fillna(0)
    def encode_units(x):  
        if x <= 0:
            return 0
        if x > 0:
            return 1
    df_for_apriori = df_for_apriori.applymap(encode_units)
    print(df_for_apriori.head(5))


    def printRules(rules):
        for index, row in rules.iterrows():
            t1 = tuple(row['antecedents'])
            t2 = tuple(row['consequents'])
            print("%s: %s ⇒ %s (support = %f, confidence = %f )"%(index, t1,t2,row['support'],row['confidence']))


    frequent_itemsets = apriori(df_for_apriori, min_support=0.004, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)
    print(frequent_itemsets.sort_values('support', ascending=False).head(10))
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
    rules = association_rules(frequent_itemsets, metric ='confidence', min_threshold = 0.00001)
    rules = rules.sort_values(by=['lift'], ascending=False).reset_index(drop=True)
    rules = rules.drop(['leverage','conviction'],axis = 1)
    print(rules.sort_values('lift', ascending=False).head(10))

    printRules(rules)

    all_antecedents = [list(x) for x in rules['antecedents'].values]
    print(all_antecedents)
    desired_indices = [i for i in range(len(all_antecedents)) if all_antecedents[i][0]==1292220]
    print(desired_indices)
    apriori_recommendations=rules.iloc[desired_indices,].sort_values(by=['lift'],ascending=False)
    print(apriori_recommendations.head()) 
    apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]
    print(apriori_recommendations_list)
    print("Apriori Recommendations for movie id: 1292220\n")
    for i in range(5):
        print("{0}: {1} with lift of {2}".format(i+1,apriori_recommendations_list[i],apriori_recommendations.iloc[i,6]))
    apriori_single_recommendations = apriori_recommendations.iloc[[x for x in range(len(apriori_recommendations_list)) if len(apriori_recommendations_list[x])==1],]
    apriori_single_recommendations_list = [list(x) for x in apriori_single_recommendations['consequents'].values]
    print("Apriori single-movie Recommendations for movie id: 1292220\n")
    for i in range(5):
        print("{0}: {1}, with lift of {2}".format(i+1,apriori_single_recommendations_list[i][0],apriori_single_recommendations.iloc[i,6]))
