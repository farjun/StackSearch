
import hparams
from dataprocess.cleaners import englishStopWords
class XmlModel(object):
    pass

class Comment(XmlModel):
    def __init__(self, attributes):
        self.id = attributes.get('Id')
        self.PostTypeId = attributes.get('PostId')
        self.AcceptedAnswerId = attributes.get('Score')
        self.CreationDate = attributes.get('Text')
        self.Score = attributes.get('CreationDate')

    def __str__(self) -> str:
        return "Comment( id = " + str(self.id) + ")"

class Post(XmlModel):
    def __init__(self, attributes, answersAttributees = None):
        self.comments = list()
        self.id = attributes.get('Id')
        self.acceptedAnswerId = attributes.get('AcceptedAnswerId')
        self.creationDate = attributes.get('CreationDate')
        self.score = attributes.get('Score')
        self.viewCount = attributes.get('ViewCount')
        self.body : str = attributes.get('Body')
        self.tags = attributes.get('Tags')
        self.answerCount = attributes.get('AnswerCount')
        self.commentCount = attributes.get('CommentCount')
        self.favoriteCount = attributes.get('FavoriteCount')
        self.title = attributes.get('Title')
        self.answers = [Answer(attr) for attr in answersAttributees] if answersAttributees else None

    def __str__(self) -> str:
        return "Post( id = {id}, answers = {answers} )".format(id=self.id, answers = self.answers)

    def __repr__(self) -> str:
        return self.__str__()

    def addComment(self, comment: Comment):
        self.comments.append(comment)

    def addAnswer(self, answer):
        self.answers.append(answer)

    def getAcceptedAnswer(self):
        if self.answers:
            for ans in self.answers:
                if ans.id == self.acceptedAnswerId:
                    return ans

        return None

    def toWordsArray(self, limit = hparams.HParams.MAX_SENTENCE_DIM):
        words = list()
        for w in self.title.split():
            if w not in englishStopWords:
                if len(words) == limit:
                    break
                words.append(w)
        return words

class Answer(Post):
    def __init__(self, attributes):
        super().__init__(attributes)

    def __str__(self) -> str:
        return "Answer( id = {id})".format(id=self.id)

    def __repr__(self) -> str:
        return self.__str__()