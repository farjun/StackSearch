import xml.etree.cElementTree as etree

from dataprocess.models import Post


class XmlParser(object):
    def __init__(self, postsFilePath, CommentsFilePath = None):
        self.CommentsFilePath = CommentsFilePath
        self.postsFilePath = postsFilePath


    def __iter__(self):
        for event, element in etree.iterparse(self.postsFilePath):
            if element.tag == "row":
                yield Post(element.attrib)


xmlParser = XmlParser("data\Comments.xml")
for post in xmlParser:
    print(post)
