
"""
This is the web-scraping file for collecting the broadway grosses data from Playbill.com
ranging from 1985 - 2019
"""

"""
getDates()
gets all weeks ending dates available for broadway grosses from playbill
writes the output into a .txt file
and returns a list
"""

def getDates():
    import urllib.request
    from bs4 import BeautifulSoup

    req = urllib.request.Request("http://www.playbill.com/grosses?week=1985-11-17")
    page = urllib.request.urlopen(req)
    html_doc = page.read()

    # parse the html page
    soup = BeautifulSoup(html_doc, "lxml")
    temp_table = soup.findAll('option')

    dateList = list()

    with open("date.txt", "w+") as f:
        for element in temp_table:
            date = element.text.strip()
            f.write(date + "\n")
            dateList.append(date)

    return dateList


"""
takes a list of dates
and writes them into separate txt files
"""

def splitDateFiles(dateList):
    for date in dateList:
        year = date.split("-", 1)[0]
        with open("weeks_in_%s.txt"%year, "a+") as f:
            f.write(date + "\n")


"""
readInDates reads in a text file of dates
and returns a list of them
"""
def readInDates(fileName):
    with open(fileName, "r") as f:
        myDateList = [line.strip() for line in f.readlines()]
    return myDateList


"""
parsePage takes in a list of week-ending dates
gets and parses associated weekely broadway grosses pages from Playbill
writes data into a csv file
"""
def parsePage(myDateList):
    import urllib.request
    from bs4 import BeautifulSoup
    import csv

    # initiate the base url
    base_URL = "http://www.playbill.com/grosses"

    with open("broadway_grosses.csv", "a+") as f:
        table_writer = csv.writer(f, delimiter=',')

        # loop over the date list and get the page associated with that date
        # write the data into the csv file
        for i in range(len(myDateList)):

            myDate = myDateList[i]
            values = {'week': myDate}
            results = urllib.parse.urlencode(values)
            results = results.encode("utf-8")

            req = urllib.request.Request(base_URL, results)
            page = urllib.request.urlopen(req)
            html_doc = page.read()

            # parse the html page
            soup = BeautifulSoup(html_doc, "lxml")
            temp_table = soup.findAll('table')[0]
            temp_all_tr = temp_table.findAll("tr")

            # write table rows into a csv file
            for nextTR in temp_all_tr:
                        csvRow = [myDate]
                        for nextTD in nextTR.findAll('td'):
                            text = nextTD.text.strip().split("\n", 1)[0]
                            csvRow.append(text)
                        table_writer.writerow(csvRow)


if __name__ == "__main__":  # execute only if run as a script  main(sys.argv)

    import csv
    # with open("broadway_grosses.csv", "a+") as f:
    #     table_writer = csv.writer(f, delimiter=',')
    #     table_writer.writerow(
    #         ['week_ending', 'show', 'this_week_gross', 'diff_in_dollars', 'avg_ticket_price', 'seats_sold', 'perfs', 'percent_of_cap', 'diff_percent_of_cap'])

    for i in reversed(range(2006, 2017)):
        dateFileName = "dates/weeks_in_%s.txt"%i
        myDateList = readInDates(dateFileName)
        parsePage(myDateList)