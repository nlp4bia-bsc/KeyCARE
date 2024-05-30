class Keyword:
    def __init__(self, text, extraction_method=None, ini=None, fin=None, score=None, label=None, categorization_method=None):
        """
        Initializes a Keyword object with the provided attributes.

        Parameters:
        text (str): The extracted keyword text. It must be provided.
        extraction_method (str): The extraction method used to find the keyword. Default: None.
        ini (int): The starting index of the keyword in the input text. Default: None.
        fin (int): The ending index of the keyword in the input text. Default: None.
        score (float): The relevance score of the keyword. Default: None.
        label (list): List of labels associated with the mention. Default: None.
        categorization_method (str): The categorization menthod used to categorize the keyword. Default: None.
        """
        self.text = text
        self.extraction_method = extraction_method
        self.score = score
        self.span = [ini, fin]
        self.label = label
        self.categorization_method = categorization_method

    def __repr__(self):
        """
        Returns a string representation of the Keyword object.

        Returns:
        str: A string representing the Keyword object.
        """
        return f"<Keyword(text='{self.text}', span='{self.span}', extraction method='{self.extraction_method}', score='{self.score}', categorization method='{self.categorization_method}', class='{self.label}')>"


class Relation:
    def __init__(self, source, target, rel_type, relation_method):
        """
        Initializes the Relation class.

        Parameters:
        source (Keyword): Source mention for the relation.
        target (Keyword): Target mention for the relation.
        rel_type (str): Type of relation.
        relation_method (str): Method used to compute the relation.
        """
        self.source = source
        self.target = target
        self.rel_type = rel_type
        self.relation_method = relation_method

    def __repr__(self):
        """
        String representation of Relation object.

        Returns:
        str: Representation of the Relation object.
        """
        return f"<Relation(source mention='{self.source.text}', target mention='{self.target.text}', relation type='{self.rel_type}', relation method='{self.relation_method}')>"
