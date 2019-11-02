import pandas as pd
import numpy as np
from datetime import datetime
import string
import re
import warnings
warnings.filterwarnings("ignore")
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from sklearn import metrics
import pprint

def clean_data(gross, social):
    #In this function, we modify the datatype for further classfication 
    #Input the gross and social datasets 
    #output the cleaned datasets

    #modify data type
    social['Date'] = pd.to_datetime(social['Date']).dt.strftime('%Y-%m-%d')
    gross['week_ending'] = pd.to_datetime(gross['week_ending']).dt.strftime('%Y-%m-%d')
    gross['label'] = [1 if x >= 0 else 0 for x in gross['diff_in_dollars']]
    social['Show'] = [re.sub(r'\s+', ' ', x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower().strip()) for x in social['Show']]
    #ain't too proud has weird special characters
    gross['show'] = [x.split('¡')[0] for x in gross['show']]
    #beautiful carol king has different names 
    gross['show'] = [re.sub(r'\s+', ' ', x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower().strip()) for x in gross['show']]
    gross.loc[gross['show']=='beautiful the carole king musical', 'show']  = 'beautiful'
    return(gross, social)

def preprocess(gross, social):
    #In this function, we generate the combined datasets from cleaned social and gross datasets
    #Input the cleaned social and gross datasets, output combined dataset

    #Match the dates from two datasets
    gross_date = [x for x in gross['week_ending'].unique() if x[0:4] in ['2019', '2018', '2017']]
    temp_date = [x.date().strftime('%Y-%m-%d') for x in list(pd.to_datetime(gross_date) + pd.Timedelta(1, unit='d'))]
    gross_date = gross_date + temp_date #+ temp_date2

    names = list(social.columns)
    names.append('label')
    df = pd.DataFrame(columns = names)

    for i in social['Date'].unique():
        temp = (pd.to_datetime(i)-pd.Timedelta(7, unit='d')).date().strftime('%Y-%m-%d')
        if temp in gross_date:
            temp_df = social.loc[social['Date'] == temp,:]
            temp_df['label'] = np.nan
            c = gross.loc[(gross['week_ending'] == temp), ['week_ending', 'show', 'label']]
            for j in temp_df.Show:
                for k in c.show:
                    if j in k:
                        temp_df.loc[(temp_df['Show'] == j),'label'] = c.loc[c['show'] == k, 'label'].values
                    elif k in j:
                        temp_df.loc[(temp_df['Show'] == j),'label'] = c.loc[c['show'] == k, 'label'].values         
        df = df.append(temp_df,ignore_index = True)
    df_notnull = df.dropna()
    return(df_notnull)

def build_model(df):
    #This function performs the Decision tree, random forest and KNN models 
    #Input the combined dataset and print out the model performance 

    X = df.loc[:, :'FB Checkins'].reset_index(drop=True)
    y = df.loc[:, 'label'].reset_index(drop=True).values
    #convert categorical data to numerical 
    for i in X.columns:
        if type(X[i][0]) == str:
            X[i] = X[i].astype("category").cat.codes
    #normalize the data   
    normDF = preprocessing.normalize(X, axis=0) 
     
    X_train, X_validate, Y_train, Y_validate = train_test_split(normDF, y, test_size=0.2, random_state=42)

    #Decision Tree and KNN models
    models = []
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))   

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits= 10, random_state=42, shuffle=False)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print("Training", msg)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validate)
        result = "%s: %f " % (name, accuracy_score(Y_validate, predictions))
        print("Testing", result)

    #Random Forest Model, generating the feature importance plot
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validate = sc.transform(X_validate)

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_validate)

    print("Random Forest Model")
    print(confusion_matrix(Y_validate,y_pred.round()))
    print(classification_report(Y_validate,y_pred.round()))
    print('Accuracy score of testing data: ', accuracy_score(Y_validate, y_pred.round()))

    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title("Feature Importance, RF")
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    feat_importances.plot(kind='barh')
    plt.show()
    return 

if __name__ == "__main__":
    gross = pd.read_csv("part2cleanedGrosses.csv", index_col=0, encoding='latin-1')
    social = pd.read_csv("part2cleanedSocialMedia.csv", index_col=0, encoding='latin-1')
    cleanedGross, cleanedSocial = clean_data(gross, social)
    combinedDf =  preprocess(cleanedGross, cleanedSocial)
    build_model(combinedDf)