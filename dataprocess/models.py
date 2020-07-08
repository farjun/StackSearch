
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
        self.PostTypeId = attributes.get('PostTypeId')
        self.AcceptedAnswerId = attributes.get('AcceptedAnswerId')
        self.CreationDate = attributes.get('CreationDate')
        self.Score = attributes.get('Score')
        self.ViewCount = attributes.get('ViewCount')
        self.Body = attributes.get('Body')
        self.Tags = attributes.get('Tags')
        self.Tags = attributes.get('AnswerCount')
        self.Tags = attributes.get('CommentCount')
        self.Tags = attributes.get('FavoriteCount')
        self.Tags = attributes.get('Tags')

    def __str__(self) -> str:
        return "Post( id = " + str(self.id) + ")"

    def addComment(self, comment: Comment):
        self.comments.append(comment)