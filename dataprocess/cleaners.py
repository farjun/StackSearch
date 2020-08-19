from nltk.corpus import stopwords
from io import StringIO
from html.parser import HTMLParser
import re
englishStopWords = set(stopwords.words('english'))

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

    @staticmethod
    def strip_tags(html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()

def cleanString(toClean):
    toClean = MLStripper.strip_tags(toClean)
    toClean = re.sub("[^a-zA-Z0-9 \n]+", "", toClean).lower()
    return toClean

def cleanQuery(toClean):
    toClean = re.sub("[^a-zA-Z0-9 \n]+", "", toClean).lower()
    toCleanSplitted = [word for word in toClean.split() if word not in englishStopWords]
    return toCleanSplitted