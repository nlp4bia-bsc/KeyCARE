class Extractor:
    def __init__(self, language, max_tokens):
        """
        Initializes an Extractor object with specified parameters.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        None
        """
        self.language = language
        self.max_tokens = max_tokens
        pass

    def extract_terms(self, text):
        """
        This method must be implemented in a subclass to extract terms from the given text.

        Parameters:
        text (str): Input text for term extraction.

        Raises:
        NotImplementedError: When called directly from the base class.

        Returns:
        None
        """
        raise NotImplementedError("extract_terms method must be implemented in subclass")

    def extract_terms_with_span(self, text):
        """
        Extracts terms from the given text and associates them with their spans.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms with spans.
        """
        terms = self.extract_terms(text)
        terms_with_span = self.find_term_span(text, terms)
        return terms_with_span

    def extract_terms_without_overlaps(self, text):
        """
        Extracts terms from the given text with their span and removing complete overlaps.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms without overlaps.
        """
        terms_with_span = self.extract_terms_with_span(text)
        terms_without_overlaps = self.rmv_overlaps(terms_with_span)
        return terms_without_overlaps

    def postprocess_terms(self, terms):
        """
        Post-processes the extracted terms if needed.

        Parameters:
        terms (list): List of extracted terms.

        Returns:
        None
        """
        pass

    @staticmethod
    def find_term_span(text, terms):
        """
        Finds the span of each term in the text.

        Parameters:
        text (str): Input text for term extraction.
        terms (list): List of extracted terms.

        Returns:
        list: List of term spans including term, start, end and score.
        """
        spans = []
        for t, score in terms:
            term = re.escape(t)
            patron = r'\b' + term + r'\b'
            coincidencias = re.finditer(patron, text, re.IGNORECASE)
            span = [(t, coincidencia.start(), coincidencia.end() - 1, score) for coincidencia in coincidencias]
            spans.extend(span)
        return spans

    @staticmethod
    def rmv_overlaps(keywords):
        """
        Removes completely overlapping terms from the list of extracted keywords.

        Parameters:
        keywords (list): List of extracted keywords with spans and other attributes (score, method, etc).

        Returns:
        list: List of non-overlapping keywords.
        """
        ent = [kw[0] for kw in keywords]
        pos = [kw[1:3] for kw in keywords]
        score = [kw[3:] for kw in keywords]
        updated_keywords = []
        repeated_words = []
        for i in range(len(ent)):
            overlap = False
            if pos[i] in repeated_words:
                overlap = True
            else:
                repeated_words.append(pos[i])
                for k in range(len(ent)):
                    if (((pos[i][0] >= pos[k][0]) and (pos[i][1] < pos[k][1])) or ((pos[i][0] > pos[k][0]) and (pos[i][1] <= pos[k][1]))):
                        overlap = True
            if (not overlap):
                kw = (ent[i], pos[i][0], pos[i][1]) + score[i]
                updated_keywords.append(kw)
        return updated_keywords