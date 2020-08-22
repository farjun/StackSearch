import xml.etree.cElementTree as etree
from Features.FeatureExtractors import FeatureExtractor
from dataprocess.models import Post
from dataprocess.cleaners import cleanString
from hparams import HParams


class XmlParser(object):
    def __init__(self, postsFilePath, CommentsFilePath = None, maxNumOfSamples = HParams.DATASET_SIZE):
        self.maxNumOfSamples = maxNumOfSamples
        self.CommentsFilePath = CommentsFilePath
        self.postsFilePath = postsFilePath

    def preproccessAttributes(self, post: Post):
        post.body = cleanString(post.body)
        post.title = cleanString(post.title)
        return post

    def getSentsGenerator(self):
        postsIter = iter(self)

        def gen():
            for post in postsIter:
                res = post.body.split()
                yield res

        return gen

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

    def getTitleGenerator(self, featureExtractor : FeatureExtractor = None):
        postsIter = iter(self)

        def gen():
            for post in postsIter:
                res = post.title.split()
                if featureExtractor:
                    res = featureExtractor.get_feature_batch(res)

                yield res

        return gen

    def __iter__(self):
        postAnswers = list()
        numOfPostsTaken = 0
        for event, element in etree.iterparse(self.postsFilePath):
            if element.tag == "row":
                if element.attrib.get('PostTypeId') == '2':
                    postAnswers.append(element.attrib)
                else:
                    res = self.preproccessAttributes(Post(element.attrib, postAnswers))
                    numOfPostsTaken+=1
                    yield res
                    if numOfPostsTaken == self.maxNumOfSamples:
                        break
                    postAnswers = list()

