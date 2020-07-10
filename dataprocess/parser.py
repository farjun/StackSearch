import xml.etree.cElementTree as etree

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
        return post

    def __iter__(self):
        for event, element in etree.iterparse(self.postsFilePath):
            if element.tag == "row":
                yield self.preproccessAttributes(Post(element.attrib))


xmlParser = XmlParser("data\Posts.xml")
for post in xmlParser:
    print(post)
