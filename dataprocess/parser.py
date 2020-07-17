import xml.etree.cElementTree as etree
import re
from Features.FeatureExtractors import FeatureExtractor
from dataprocess.models import Post
from io import StringIO
from html.parser import HTMLParser

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


class XmlParser(object):
    def __init__(self, postsFilePath, CommentsFilePath = None):
        self.CommentsFilePath = CommentsFilePath
        self.postsFilePath = postsFilePath


    def preproccessAttributes(self, post: Post):
        post.body = MLStripper.strip_tags(post.body)
        post.body = re.sub("[^a-zA-Z0-9 \n]+", "", post.body)
        return post

    def getWordsGenerator(self, featureExtractor : FeatureExtractor = None):
        postsIter = iter(self)

        def gen():
            for post in postsIter:
                res = post.body.split()
                if featureExtractor:
                    res = featureExtractor.get_feature_batch(res)
                for word in res:
                    yield word

        return gen

    def __iter__(self):
        postAnswers = list()
        for event, element in etree.iterparse(self.postsFilePath):
            if element.tag == "row":
                if element.attrib.get('PostTypeId') == '2':
                    postAnswers.append(element.attrib)
                else:
                    res = self.preproccessAttributes(Post(element.attrib, postAnswers))
                    yield res

                    postAnswers = list()

