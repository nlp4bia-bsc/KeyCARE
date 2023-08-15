class Extractor:
    def __init__(self, language, max_tokens):
        self.language = language
        self.max_tokens = max_tokens
        pass

    def extract_terms(self, text):
        raise NotImplementedError("extract_terms method must be implemented in subclass")

    def extract_terms_with_span(self, text):
        terms = self.extract_terms(text)
        terms_with_span = self.find_term_span(text, terms)
        return terms_with_span

    def extract_terms_without_overlaps(self, text):
        terms_with_span = self.extract_terms_with_span(text)
        terms_without_overlaps = self.rmv_overlaps(terms_with_span)
        return terms_without_overlaps

    def postprocess_terms(self, terms):
        pass

    @staticmethod
    def find_term_span(text, terms):
        spans = []
        for t,score in terms:
          term = re.escape(t)
          patron = r'\b' + term + r'\b'
          coincidencias = re.finditer(patron, text, re.IGNORECASE)
          span = [(t, coincidencia.start(), coincidencia.end()-1, score) for coincidencia in coincidencias]
          spans.extend(span)
        return spans

    @staticmethod
    def rmv_overlaps(keywords):
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
          kw = (ent[i],pos[i][0],pos[i][1]) + score[i]
          updated_keywords.append(kw)
      return updated_keywords