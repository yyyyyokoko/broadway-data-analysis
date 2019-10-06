import warnings
warnings.filterwarnings('ignore')

import urllib
import urllib.parse
import urllib.request
import re
import requests
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def readData(fileName):
    #read the show names and clean up the unformatted names
    #return an empty dataset to store further data
    musical_list = pd.read_csv(fileName)
    musical_list.columns = ['show']
    musical_list["total_rating"] = np.nan
    musical_list["critics_rating"] = np.nan
    musical_list["readers_rating"] = np.nan
    for i in range(len(musical_list)):
        text = musical_list["show"][i]
        musical_list["show"][i] = text.split("?", 1)[0]

    return musical_list

def getReviews(musical_list):
    #scape reviews(text) and ratings(numerical) data from the webpage
    #For those the review webpage exists,
    #return two dataframes, one contains the three different ratings of each show
    #the other one contains the different text reviews from critics of each show

    url_base = 'https://www.broadwayworld.com/reviews/' # Define URL base
    total_review = pd.DataFrame(columns = ["show", "rating", "title", "content"]) 
    for i in range(len(musical_list["show"])): 
        #modify the show names for url links
        query = "".join((char if char.isalpha() else " ") for char in musical_list["show"][i]).split()
        query = "-".join(query)
        url = url_base + query
        r = requests.get(url) #scrapping webpage
        html_doc = r.text
        soup = BeautifulSoup(html_doc, 'html.parser')
        context = soup.findAll('div', {'class':'context'})  #finding keywords to get text
        rating = soup.findAll('div', {'class':'rating-box'})
        if len(rating) == 1: #opt-out invalid page
            temp = pd.DataFrame(columns = ["show", "rating", "title", "content"]) #get reviews 
            values = context[0].find_all('div', {'class':['titles', 'score', 'title-block', 'text']})
            cleaned_values = []
            for p in range(len(values)):
                cleaned_values.append(values[p].get_text())

            #put data into correct column 
            for j in range(len(cleaned_values)):
                temp.loc[j, 'show'] = musical_list["show"][i]
                if j % 3 == 0:
                    temp.loc[j, 'rating'] = int(cleaned_values[j])
                elif j % 3 == 1:
                    temp.loc[j-1, 'title'] = cleaned_values[j]
                elif j % 3 == 2:
                    temp.loc[j-2, 'content'] = cleaned_values[j]
            temp = temp.dropna()
            temp = temp.reset_index(drop=True)
            total_review = total_review.append(temp)

            scores = rating[0].find_all('div', {'class':'score'}) #get ratings
            for k in range(len(scores)):
                musical_list.iloc[i, k+1] = scores[k].get_text()
            print(musical_list["show"][i], "Retriving reviews success") #success massage 
        else:
            pass
    return musical_list, total_review

def dataCleaning(musical_list, total_review):
    #basic cleaning to make dataset has valid values and correct dimensions 
    #return three dataframes: the rating with NA rows, the rating without NA rows, and cleaned reviews 

    #remove NA values 
    musical_list_news =  musical_list.reset_index(drop=True)
    musical_list1 = musical_list.dropna(thresh=3).reset_index(drop=True)

    clean_review = total_review.reset_index(drop=True)
    clean_review['company'],  clean_review['author'] ,clean_review['date']  = np.nan, np.nan, np.nan

    #split one column to four different columns since it contained multiple information
    for i in range(len(clean_review.title)):
        temp = clean_review.title[i].split('From:')
        #print(temp)
        clean_review['title'][i] = temp[0].strip()
        a = temp[1].split("|")
        clean_review['company'][i] = a[0].strip()
        clean_review['author'][i] = a[1].split(":")[-1].strip()
        clean_review['date'][i] = a[2].split(":")[-1].strip()
        clean_review["content"][i] = clean_review["content"][i].strip()
    return musical_list_news, musical_list1, clean_review

def getLinks(show):
    #the news link for individual show is under a search page 
    #get links for individual show
    #return 5 links in a list
    news = 'http://www.playbill.com/searchpage/search?q='+show+'&sort=Relevance&articles=on&qasset=' 
    r = requests.get(news)
    html_doc = r.content
    soup = BeautifulSoup(html_doc, 'html.parser') 
    tag = "/article" 
    first =soup.findAll('div', {'class':"bsp-list-promo-title"}) #find correct links 
    base_url = "http://www.playbill.com"
    links = []
    e = 0
    while len(links) < 5 and len(first) != 0:
        j = first[e].findAll('a', attrs={'href': re.compile(tag)})
        if len(j) == 1:
            link = base_url + j[0].get('href')
            links.append(link)
        e += 1
        if e > len(first)-1:        
            break
    return links

def getNews(musical_list_news):
    #retrieve news for each show, return the a dataframe contained all news scrapped
    allNews = pd.DataFrame(columns = ["show", "title", "subtitle", "content"])
    for show in musical_list_news['show']:
        df_news = pd.DataFrame(columns = ["show", "title", "subtitle", "content"])
        showname = show.replace(" ", "+") #modify for urls 
        links = getLinks(showname) #get links 

        if len(links) == 5: #we want at least 5 news for each show
            for i in range(len(links)):
                r = requests.get(links[i])
                html_doc = r.content
                soup = BeautifulSoup(html_doc, 'html.parser')
                titles= soup.findAll('span', {'class':["heading-one bsp-article-title"]})
                subtiles =  soup.findAll('span', {'class':"heading-three bsp-article-subtitle" })

                #get title and substitle of one news 
                if len(subtiles) == 1 & len(titles) == 1:
                    temp = [show, titles[0].get_text(), subtiles[0].get_text()]
                elif len(titles) == 1 & len(subtiles) == 0:
                    temp = [show, titles[0].get_text(), np.nan]
                else:
                    temp = [show, np.nan, np.nan]
                com = soup.findAll('div', {'class':"bsp-article-content"})
                #get content 
                context_temp = []
                for k in com[0].find_all('p'):
                    context_temp.append(k.get_text())
                context_temp = "".join(context_temp)
                temp.append(context_temp.strip())
                df_news.loc[i] = temp
            print(show, "Scraping success")
            allNews = allNews.append(df_news)
    return allNews.reset_index(drop=True)

if __name__ == "__main__":
    musical = readData("broadway-shows-all.csv")
    musical_list, total_review = getReviews(musical)
    musical_list_news, musical_list1, clean_review = dataCleaning(musical_list, total_review)
    #musical_list_news.to_csv("Musical_ratings-withNa.csv", index = False) #save to file
    musical_list1.to_csv("Musical_ratings.csv", index = False)
    clean_review.to_csv("clean_review.csv", index = False)
    result = getNews(musical_list_news) 
    result.to_csv("news.csv", index = False)

