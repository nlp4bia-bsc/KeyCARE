class Keyword:
    def __init__(self, text, method, ini, fin, score):
        """
        Initializes a Keyword object with the provided attributes.

        Parameters:
        text (str): The extracted keyword text.
        method (str): The extraction method used to find the keyword.
        ini (int): The starting index of the keyword in the input text.
        fin (int): The ending index of the keyword in the input text.
        score (float): The relevance score of the keyword.

        Returns:
        None
        """
        self.text = text
        self.method = method
        self.score = score
        self.span = [ini, fin]
        self.label = None

    def __repr__(self):
        """
        Returns a string representation of the Keyword object.

        Parameters:
        None

        Returns:
        str: A string representing the Keyword object.
        """
        return f"<Keyword(text='{self.text}', method='{self.method}', score='{self.score}', span='{self.span}', class='{self.label}')>"
