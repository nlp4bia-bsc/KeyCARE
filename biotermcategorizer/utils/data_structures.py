class Keyword:
    def __init__(self, text, method, ini, fin, score):
        self.text = text
        self.method = method
        self.score = score
        self.span = [ini, fin]

    def __repr__(self):
        """
        A method to return a string representation of the Keyword object.

        Parameters:
        - None

        Returns:
        - A string representing the Keyword object.
        """
        return f"<Keyword(text='{self.text}', method='{self.method}', score='{self.score}', span='{self.span}')>"