
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
    def __init__(self, attributes):
        self.comments = list()

        self.id = attributes.get('Id')
        self.postTypeId = attributes.get('PostTypeId')
        self.acceptedAnswerId = attributes.get('AcceptedAnswerId')
        self.creationDate = attributes.get('CreationDate')
        self.score = attributes.get('Score')
        self.viewCount = attributes.get('ViewCount')
        self.body = attributes.get('Body')
        self.tags = attributes.get('Tags')
        self.answerCount = attributes.get('AnswerCount')
        self.commentCount = attributes.get('CommentCount')
        self.favoriteCount = attributes.get('FavoriteCount')

    def __str__(self) -> str:
        return "Post( id = " + str(self.id) + ")"

    def addComment(self, comment: Comment):
        self.comments.append(comment)