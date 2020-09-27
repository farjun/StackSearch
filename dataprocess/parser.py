import xml.etree.cElementTree as etree
from features.FeatureExtractors import FeatureExtractor
from dataprocess.models import Post
from dataprocess.cleaners import cleanString
from hparams import HParams
from gensim.models.doc2vec import TaggedDocument

class XmlParser(object):
    def __init__(self, postsFilePath, CommentsFilePath = None, trainDs = True, parseRange = None):
        datasetRange = HParams.TRAIN_DATASET_RANGE if trainDs else HParams.TEST_DATASET_RANGE
        self.minNumOfSamples = datasetRange[0]
        self.maxNumOfSamples = datasetRange[1]
        if parseRange:
            self.minNumOfSamples = parseRange[0]
            self.maxNumOfSamples = parseRange[1]

        self.CommentsFilePath = CommentsFilePath
        self.postsFilePath = postsFilePath
        self.titlesCache = dict()

    def preproccessAttributes(self, post: Post):
        post.body = cleanString(post.body)
        post.title = cleanString(post.title)
        return post

    def getPostTitle(self, postId):
        return self.titlesCache.get(postId)

    def getSentsGenerator(self, tagged=False):
        postsIter = iter(self)

        def gen():
            for i, post in enumerate(postsIter):
                title_tokens = post.title.split()
                if not tagged:
                    res = title_tokens
                else:
                    res = TaggedDocument(title_tokens, [i])
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
        def gen():
            postsIter = iter(self)
            for post in postsIter:
                res = post.toWordsArray()
                if len(res) == 0:
                    continue
                if featureExtractor:
                    res = featureExtractor.get_feature_batch(res)
                self.titlesCache[post.id] = post.toWordsArray()
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
                    numOfPostsTaken += 1
                    if numOfPostsTaken < self.minNumOfSamples:
                        continue
                    res = self.preproccessAttributes(Post(element.attrib, postAnswers))
                    yield res
                    if numOfPostsTaken == self.maxNumOfSamples:
                        break
                    postAnswers = list()

