import os
import re
import nltk
nltk.download('stopwords')

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

    def extract_terms_without_overlaps(self, text, kwargs):
        """
        Extracts terms from the given text with their span and removing complete overlaps.

        Parameters:
        text (str): Input text for term extraction.
        **kwargs: Additional arguments for training. If some of the kwargs for the extraction do not exist, they will be ignored.

        Returns:
        list: List of extracted terms without overlaps.
        """
        self.kwargs = kwargs
        terms_with_span = self.extract_terms_with_span(text)
        terms_without_overlaps = self.rmv_overlaps(terms_with_span)
        return terms_without_overlaps

    def postprocess_terms(self, terms):
        """
        Post-processes the extracted terms if needed by removing and modifying some terms in a specific order.

        Parameters:
        terms (list): List of extracted terms.

        Returns:
        new_terms (list): List of extracted terms after postprocessing.
        """
        new_terms = self.rmv_meaningless(terms)
        new_terms = self.rmv_stopwords(new_terms)
        new_terms = self.rmv_stopwords(new_terms)
        new_terms = self.rmv_nonalnum_characters(new_terms)
        new_terms = self.rmv_nonalnum_characters(new_terms)
        new_terms = self.rmv_meaningless(new_terms)
        return new_terms

    def rmv_meaningless(self, terms):
        """
        Removes meaningless terms from the list of terms (numbers, terms with just one character and stopwords).

        Parameters:
        terms (list): List of terms to be filtered.

        Returns:
        list: List of terms with meaningless terms removed.
        """
        self.stopwords = nltk.corpus.stopwords.words(self.language)
        new_terms = [t for t in terms if (t[0] not in self.stopwords and len(t[0]) >= 2 and not t[0].isdigit())]
        return new_terms

    def rmv_stopwords(self, terms):
        """
        Removes stopwords at the start and end of terms from the list.

        Parameters:
        terms (list): List of terms to be filtered.

        Returns:
        list: List of terms with stopwords removed.
        """
        new_terms = terms
        self.stopwords = nltk.corpus.stopwords.words(self.language)
        for word in self.stopwords:
            new_terms = [(t[0][(len(word)+1):], t[1] + len(word) + 1, t[2], t[3], t[4]) if (t[0].lower().startswith(word.lower() + " ")) else t for t in new_terms]
            new_terms = [(t[0][:-(len(word)+1)], t[1], t[2] - len(word) - 1, t[3], t[4]) if (t[0].lower().endswith(" " + word.lower())) else t for t in new_terms]
        return new_terms

    def rmv_nonalnum_characters(self, terms):
        """
        Removes non-alphanumeric characters at the start and end of the terms.

        Parameters:
        terms (list): List of terms to be filtered.

        Returns:
        list: List of terms with non-alphanumeric characters removed.
        """
        new_terms = [(t[0][1:], t[1] + 1, t[2], t[3], t[4]) if not t[0][0].isalnum() else t for t in terms]
        new_terms = [(t[0][:-1], t[1], t[2] - 1, t[3], t[4]) if not t[0][-1].isalnum() else t for t in new_terms]
        return new_terms

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
            span = [(t, coincidencia.start(), coincidencia.end(), score) for coincidencia in coincidencias]
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