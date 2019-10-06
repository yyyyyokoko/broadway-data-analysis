import csv
import re
import requests
from bs4 import BeautifulSoup

names = []

with open("broadway-shows-all.csv") as f:

    file = csv.reader(f)
    for row in file:
        for item in row:
            names.append(item)


def get_clean_broadway_wiki(show_name):

    show_name = show_name.replace(" ", "_")

    url = "https://en.wikipedia.org/wiki/" + show_name
    response = requests.get(url)
    txt = response.text
    text = BeautifulSoup(txt, 'lxml')
    text = str(text)

    # Remove XML except hyperlinks and headings
    text = text.replace("&gt;",">").replace("&lt;", "<").replace("&amp;", "&").replace("&nbsp;", " ")
    text = re.sub(r"<(/a|a [^<>]*)>",r"#@Q\1Q@#", text)
    text = re.sub(r"<(/?h[0-9])>", r"#@Q\1Q@#", text)
    text = re.sub('<[^<>]+>', '', text)
    text = text.replace("#@Q", "<").replace("Q@#", ">")

    # Mark up
    text = re.sub(r'<h[0-9]>([^\n<>]+)</h[0-9]>', r"<head>\1</head>", text)
    text = re.sub(r'(<a href)', r"<ref target", text)
    text = re.sub(r'(target="[^"]+)%3A', r"\1:", text)
    text = re.sub(r'(</?)a([ >])', r"\1ref\2", text)
    text = re.sub(r'BULLET::::-?([^\n]+)', r"<item>\1</item>", text)
    text = re.sub(r'^([^<\s][^\n]*)$', r'<p>\1</p>', text,flags=re.MULTILINE)
    text = re.sub(r'♡❤♡([^❤♡]+)♡❤♡', r'<hi rend="bold">\1</hi>', text)
    text = re.sub(r'♡❤([^❤♡]+)♡❤', r'<hi rend="italic">\1</hi>', text)
    text = re.sub(r' +', r" ", text)

    pats = [(r'((<item>.*?</item>\n?)+)', r'<list type="unordered">\n\1</list>\n')]

    text = text.replace("&quot;", '"')
    for pat in pats:
        text = re.sub(pat[0], pat[1], text, flags=re.MULTILINE|re.DOTALL)

    return text


for name in names[1:]:
    if '/' not in name:
        with open(name + ".xml", 'w') as f:
            file = f.write(get_clean_broadway_wiki(name))
