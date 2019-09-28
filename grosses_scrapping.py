def getQuote():
    import urllib.request
    from bs4 import BeautifulSoup

    myURL = "http://finance.yahoo.com/quote/"
    stop = False
    while (stop == False):
        ticker = input("Enter your stock ticker (enter 'stop' to end the program): ")
        if ticker != "stop":
            full_url = myURL + ticker
            req = urllib.request.Request(full_url)
            page = urllib.request.urlopen(req)
            soup = BeautifulSoup(page, "html.parser")
            price = soup.find("span", {"Trsdu(0.3s) Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(b)"}).text
            print(price)
        else:
            print("getQuote() teminated")
            stop = True

def date():
    return None