# Loading Libraries
import pandas as pd
import string
import re
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def clean_data(gross):
    """
    In this function, we modify the datatype for further classification tasks
    :param gross: the Broadway Grosses Data Set
    :return: the cleaned data sets
    """

    # modify data type
    gross['week_ending'] = pd.to_datetime(gross['week_ending']).dt.strftime('%Y-%m-%d')
    gross['label'] = [1 if x >= 0 else 0 for x in gross['diff_in_dollars']]
    # ain't too proud has weird special characters
    gross['show'] = [x.split('¡')[0] for x in gross['show']]
    # beautiful carol king has different names
    gross['show'] = [re.sub(r'\s+', ' ', x.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower().strip()) for x in gross['show']]
    gross.loc[gross['show'] == 'beautiful the carole king musical', 'show'] = 'beautiful'

    return gross


def convert_attribute(df):

    """
    This function converts one attribute into categorical labels based on the binning results and adds this label
    to the dataframe for the classification task
    :param df: the Broadway Grosses Data Set
    :return: an updated dataframe that has the label information
    """

    df['Popularity'] = ['Oversold' if x == '(1.0, 2.0]' else 'Unpopular' if x == '(-1.0, 0.0]' or x == '(0.0, 0.5]' else 'Popular' for x in df['percent_of_cap_binRange1']]

    return df


def naive_bayes(df):

    """
    This function performs the Naive Bayes classifier on the classification task
    :param df: the updated dataframe from the function convert_attribute()
    :return: the model performance
    """

    X = df.loc[:, :'perfs'].reset_index(drop=True)
    y = df.loc[:, 'Popularity'].reset_index(drop=True).values

    # convert categorical data to numerical
    for i in X.columns:
        if type(X[i][0]) == str:
            X[i] = X[i].astype("category").cat.codes
    # normalize the data
    normDF = preprocessing.normalize(X, axis=0)

    X_train, X_validate, Y_train, Y_validate = train_test_split(normDF, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_validate)

    kfold = KFold(n_splits=10, random_state=42, shuffle=False)
    cv_results = cross_val_score(gnb, X_train, Y_train, cv=kfold, scoring='accuracy')
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())

    print("Confusion Matrix")
    print(confusion_matrix(Y_validate, y_pred))
    print(classification_report(Y_validate, y_pred))
    print("Accuracy score of training data: ", msg)
    print('Accuracy score of testing data: ', accuracy_score(Y_validate, y_pred))
    print("")

    return


if __name__ == "__main__":

    myData = pd.read_csv('part2cleanedGrosses.csv', sep=',', encoding='latin1')
    convertedDF = convert_attribute(clean_data(myData))
    nbclf = naive_bayes(convertedDF)

    print(convertedDF["Popularity"].value_counts())