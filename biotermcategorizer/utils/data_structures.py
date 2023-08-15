class Keyword:
    def __init__(self, text, method, ini, fin, score):
        self.text = text
        self.method = method
        self.score = score
        self.span = [ini, fin]