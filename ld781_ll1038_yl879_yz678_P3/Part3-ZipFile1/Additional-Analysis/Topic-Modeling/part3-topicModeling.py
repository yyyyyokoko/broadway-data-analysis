import pandas as pd
import warnings
from pprint import pprint
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
import string, re
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import gensim, spacy, logging
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS


def preprocess(review):
    """
    In this function, we cleaned the textual review data for further topic modeling task
    :param gross: the Broadway Review Data Set
    :return: the cleaned data sets and a list of stop words
    """
    review.content = review.title + '. ' + review.content
    review = review.drop(columns=['title'])
    # Remove punctuation
    review['content'] = review['content'].str.replace('[^\w\s]','')
    review['content'] = review['content'].str.lower()
    content = review['content'].dropna().tolist()

    stop_words = stopwords.words('english')
    stop_words.extend(['broadway', 'review', 'play', 'theatre', 'musical', 'theater', 'show', 'like', 'production',
                       'one', 'performance'])

    data_words = []
    for sent in content:
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        data_words.append(sent)
    
    return data_words, stop_words


def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    In this function, we preprocess the textual review data for further topic modeling task
    :param texts: the cleaned Broadway Review Data Set
    :param stop_words: created stop words
    :param allowed_postags: a set of parameters for LDA model
    :return: the processed data set and the original data set
    """

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out, texts


def modeling(data_ready, texts):
    """
    In this function, we build the LDA model
    :param data_ready: the preprocessed Broadway Review Data Set
    :param texts: the original review data set
    :return: LDA model and topics
    """

    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, random_state=100,
                                                update_every=1, chunksize=10, passes=10, alpha='symmetric',
                                                iterations=100, per_word_topics=True)

    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                                                                  topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return lda_model, df_dominant_topic


def visual(lda_model, df_dominant_topic):
    """
    In this function, we created visualizations for topic modeling 
    :param lda_model: the preprocessed Broadway Review Data Set
    :param df_dominant_topic: the original review data set
    :return: two visualizations 
    """

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):    
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=100, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 100), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.show()

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words, background_color='white', width=2500, height=1800, max_words=15,
                      colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    return 


if __name__ == "__main__":
    review = pd.read_csv("clean_review.csv")
    data_words, stop_words = preprocess(review)
    data_ready, texts = process_words(data_words, stop_words) 
    lda_model, df_dominant_topic = modeling(data_ready, texts)
    visual(lda_model, df_dominant_topic)
