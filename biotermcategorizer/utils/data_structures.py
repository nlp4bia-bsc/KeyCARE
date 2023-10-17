class Keyword:
    def __init__(self, text, extraction_method, ini, fin, score):
        """
        Initializes a Keyword object with the provided attributes.

        Parameters:
        text (str): The extracted keyword text.
        extraction_method (str): The extraction method used to find the keyword.
        ini (int): The starting index of the keyword in the input text.
        fin (int): The ending index of the keyword in the input text.
        score (float): The relevance score of the keyword.
        """
        self.text = text
        self.extraction_method = extraction_method
        self.score = score
        self.span = [ini, fin]
        self.label = None
        self.categorization_method = None

    def __repr__(self):
        """
        Returns a string representation of the Keyword object.

        Returns:
        str: A string representing the Keyword object.
        """
        return f"<Keyword(text='{self.text}', span='{self.span}', extraction method='{self.extraction_method}', score='{self.score}', categorization method='{self.categorization_method}', class='{self.label}')>"