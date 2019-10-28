from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt

# find all the show names
def find_name_list():
    url='https://www.broadwayworld.com/industry-social.cfm'
    page=requests.get(url)
    soup = BeautifulSoup(page.text, "lxml")
    # find the table which contains all the show names
    table=soup.findAll("table")[0]
    All_TR=table.findAll("tr")
    list = []
    for nextTR in All_TR:
        for nextTD in nextTR.findAll("td"):
            # find the row name of each row, which is the name of the show
            # they start with "b"s
            for nextb in nextTD.findAll('b'):
                list.append(nextb.text.strip())
    # several values are not needed, show names start from the last time "Total Fans Change" occurs
    starting_index = list.index('Total Fans Change', 100)
    list = list[starting_index + 1:]
    return list

# get the values in each page
def UseRequest(BaseURL, name):
    URLPost = {'show_name': name}
    page = requests.get(BaseURL, URLPost)
    soup = BeautifulSoup(page.text, "lxml")
    table = soup.findAll("table")[0]
    # start from the third row, because the first two are titles
    All_TR = table.findAll("tr")[2:-1]
    csvFile = open('scrap_of_social_media.csv', "a")
    playerwriter = csv.writer(csvFile, delimiter=',')
    for nextTR in All_TR:
        csvRow = [name]
        for nextTD in nextTR.findAll("td"):
            csvRow.append(nextTD.text.strip())
        playerwriter.writerow(csvRow)
    csvFile.close()


def main(argv):
    # Find the list of show name, to generate URLs to access content about each show
    name_list = find_name_list()

    # Use the list of names to scrap content in each page
    BaseURL = "https://www.broadwayworld.com/industry-social.cfm?"
    # open a file to save the data and write headers
    File = open('scrap_of_social_media.csv', "w")
    myWriter = csv.writer(File, delimiter=',')
    myWriter.writerow(['Show', 'Date', 'FB Likes', 'Likes Vs.Last Week', 'FB Talking About',
                       'Talking Vs.Last Week', 'FB Checkins', 'Checkins vs.Last Week', 'Twitter Followers',
                       'Twitter vs.Last Week', 'Instagram Followers', 'IG Followers vs.Last Week',
                       'Current', 'Type', 'Total Fans Change'])
    File.close()
    for name in name_list:
        UseRequest(BaseURL, name)


if __name__ == "__main__":
    main(sys.argv)