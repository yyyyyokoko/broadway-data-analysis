from bs4 import BeautifulSoup
import requests
import re
import urllib
baseURL = 'https://www.broadwayworld.com/industry-social.cfm'
myURL = baseURL
results=urllib.parse.urlencode(values)
# Put data in bytes
results = results.encode("utf-8")
req = urllib.request.Request(myURL, results)
# visit the site with the values
resp = urllib.request.urlopen(req)
resp_results = resp.read()
resp_results = resp_results.decode("utf-8")

name_pattern = re.compile('Trsdu\(0.3s\) Fw\(b\) Fz\(36px\) Mb\(-4px\) D\(i?b\)" data-reactid="\d*">(\d*\.\d{2})</span>')
price = re.findall(name_pattern, resp_results)
ticketPrice = price[0]
print("The current price of", ticket, "is:", ticketPrice)