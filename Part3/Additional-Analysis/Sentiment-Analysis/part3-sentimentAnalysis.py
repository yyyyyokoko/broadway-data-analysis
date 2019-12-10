import pandas as pd
import re
from string import punctuation
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import time


def split_dataset_manually_label():
    """
    In this function, we split the review data set to leave 15% of the reviews to be manually annotated
    """
    # read in the original dataset, split a 15% to manually label them as positive, negative or neutral
    df = pd.read_csv("clean_review.csv")
    df_label = df.sample(frac=0.15)
    # save the 15%
    df_label.to_csv('to_be_labeled.csv', index=False, sep=',')
    # save the residual 85% that do not have labels
    rowlist = []
    for indexs in df_label.index:
        rowlist.append(indexs)
    df_unlabeled = df.drop(rowlist, axis=0)
    df_unlabeled.to_csv('unlabeled.csv', index=False, sep=',')


def read_in_dataset(unlabeled, labeled):
    """
    In this function, we read in the unlabeled data set as training set, and manually annotated data as testing set
    :return:
    """
    df_main = pd.read_csv(unlabeled)
    df_manual = pd.read_csv(labeled)

    # convert the review scores into 3 categories
    # first explore the distribution of the score
    df_main.rating.value_counts()
    dictionary = {'0': '0', '1': '0', '2': '0', '3': '0', '4': '0', '5': '0', '6': '0', '7': '1', '8': '1', '9': '2',
                  '10': '2'}
    # convert the value into 3 categories: 0 for negative, 1 for neutral, and 2 for positive
    for key in dictionary.keys():
        df_main['rating'] = df_main['rating'].astype(str)
        df_main['rating'] = df_main['rating'].replace(key, dictionary[key])
    return df_main, df_manual


def simplify_to_binary(df_main, df_manual):
    """
    This function is used to drop the neutral ones so the model would do a modified task of binary classification
    """
    df_main = df_main.drop(df_main.loc[df_main['rating'] == '1'].index)
    df_main = df_main.reset_index(drop=True)
    df_manual = df_manual.drop(df_manual.loc[df_manual['manual_label'] == 1].index)
    df_manual = df_manual.reset_index(drop=True)
    return df_main, df_manual


def clean_content(words, stopwords_list):
    # convert words into lower-case
    words = words.lower()
    # remove numbers in the review
    words = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', words)

    # tokenization
    wlist = word_tokenize(words)

    # remove stopwords
    wlist = [word for word in wlist if word not in stopwords_list]

    # lemmatization
    # wlist = [Stemmer.stem(word) for word in wlist]
    wlist = [WordNetLemmatizer().lemmatize(word, pos="v") for word in wlist]

    return " ".join(wlist)


def data_cleaning(df):
    """
    This function together with the clean_content() function, tokenized, lemmatized corpus, and removed stop words
    """
    stopwords_broadway = ['play', 'perform', 'broadway', 'make', '``']
    stopwords_list = set(stopwords.words('english') + list(punctuation)+stopwords_broadway)
    for i in range(len(df["content"])):
        df.loc[i, "content"] = clean_content(df.loc[i, "content"], stopwords_list)
    return df


def train_dev_split(df):
    """
    This function splited the training and dev data set
    :param df: training data set
    :return: X_train, X_dev, y_train, y_dev
    """
    df = df.dropna()
    # split the dataset in train and test
    X = df['content']
    y = df['rating']
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return X_train, X_dev, y_train, y_dev


def vectorization(X_train, X_dev, X_test):
    """
    This function converts Xs to vectors
    """
    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.fit(X_train)
    X_train_dtm = vectorizer.transform(X_train)
    X_dev_dtm = vectorizer.transform(X_dev)
    X_test_dtm = vectorizer.transform(X_test)
    return X_train_dtm, X_dev_dtm, X_test_dtm


def grid_search(X_train_dtm, y_train,models):
    """
    This function find the best combination of parameters
    :param X_train_dtm, y_train: training data of prepossessed X and Y
    :param models: machine learning models with parameter lists
    :return: best combination of parameters and time cost
    """
    best = []
    time_cost = []
    for name, model, param in models:
        grid = GridSearchCV(estimator=model, param_grid=param, cv=8, n_jobs=-1, scoring='recall_micro')
        # calculate time as a dimension of evaluation
        start_time = time.time()

        # fit the model
        result = grid.fit(X_train_dtm, y_train)

        end_time = time.time()
        cost_grid_search = (end_time - start_time) / 60

        # get the best parameters
        best.append(result.best_params_)
        time_cost.append(cost_grid_search)
        print(name)
        print("best parameters:", result.best_params_)
        print("time cost in grid search (mins):", cost_grid_search)
    return best, time_cost


def model_evaluation(best,time_cost,models, X_train_dtm, y_train, X_dev_dtm, y_dev,names):
    """
    This function give the results of machine learning models with best hyperparameters
    :param best,time_cost,models: best parameters, time_cost in grid search, and models' name
    :param X_train_dtm, y_train, X_dev_dtm, y_dev: X, y in the training set, and X, y in the dev or testing set
    :param names: the row names in the result table
    """
    counter = 0
    for name, model, para_list in models:
        clf = model
        clf.set_params(**best[counter])
        # calculate time as a dimension of evaluation
        start_time = time.time()
        clf.fit(X_train_dtm, y_train)
        y_prediction = clf.predict(X_dev_dtm)

        end_time = time.time()
        cost_model_training = (end_time - start_time) / 60

        # calculate total time elapse in parameter selection and model training
        cost = cost_model_training + time_cost[counter]
        print("Total time cost was %.2f minutes" % cost)

        # print the score table
        print("Confusion_matrix:")
        print(confusion_matrix(y_true=y_dev, y_pred=y_prediction))
        print("Accuracy: ", accuracy_score(y_true=y_dev, y_pred=y_prediction))
        print("Precision, Recall, F score:")
        print(classification_report(y_true=y_dev, y_pred=y_prediction, target_names=names))

        counter += 1


if __name__ == "__main__":
    # Our input data was cleaned review data set, we splitted 15% of them to be manually annotated.
    # The following is the function to do so, but we commented it as the data set was splitted and no need to do it
    # again when you review the code

    # split_dataset_manually_label()

    # read in the splitted data sets and converted ratings into categories
    # df_main is what we used to train the model and search for the best hyperparameters
    # df_manual is the testing set which contains the manual labels
    df_main, df_manual = read_in_dataset("unlabeled.csv", "manually_annotated.csv")

    # tokenizing, lemmatizing, stemming, removing the stop words, etc.
    df_main = data_cleaning(df_main)
    df_manual = data_cleaning(df_manual)

    ### multivariate classification ###
    # train models
    X_train, X_dev, y_train, y_dev = train_dev_split(df_main)
    X_train_dtm, X_dev_dtm, X_test_dtm = vectorization(X_train, X_dev, df_manual['content'])
    # find the best hyperparameters
    models = [('lr', LogisticRegression(penalty='l2'),
               dict(dual=[True, False], max_iter=[80, 90, 100, 110], C=[0.5, 1.0, 1.5])),
              ('svm', svm.SVC(), dict(kernel=['linear', 'rbf'], C=[1, 0.25, 0.5], gamma=[0.01, 1]))]
    best, time_cost = grid_search(X_train_dtm, y_train, models)
    # model evaluation on training set
    names_multi = ['negative', 'neutral', 'positive']
    model_evaluation(best, time_cost, models, X_train_dtm, y_train, X_dev_dtm, y_dev, names_multi)
    # model evaluation on testing set
    model_evaluation(best, time_cost, models, X_train_dtm, y_train, X_test_dtm, df_manual['manual_label'].astype(str),
                     names_multi)

    ### binary classification ###
    # drop the data with neutral labels, let the models do a binary classification task to compare the results
    df_main_bi, df_manual_bi = simplify_to_binary(df_main, df_manual)

    # repeat the above steps to train models
    # train models
    X_train, X_dev, y_train, y_dev = train_dev_split(df_main_bi)
    X_train_dtm, X_dev_dtm, X_test_dtm = vectorization(X_train, X_dev, df_manual_bi['content'])

    # find the best hyperparameters
    best, time_cost = grid_search(X_train_dtm, y_train, models)
    # model evaluation on training set
    names_bi = ['negative', 'positive']
    model_evaluation(best, time_cost, models, X_train_dtm, y_train, X_dev_dtm, y_dev, names_bi)
    # model evaluation on testing set
    model_evaluation(best, time_cost, models, X_train_dtm, y_train, X_test_dtm,
                     df_manual_bi['manual_label'].astype(str), names_bi)
