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
        for t in terms:
          term = re.escape(t)
          patron = r'\b' + term + r'\b'
          coincidencias = re.finditer(patron, text, re.IGNORECASE)
          span = [[coincidencia.start(), coincidencia.end()-1] for coincidencia in coincidencias]
          spans.append((t,span))
        return spans

    @staticmethod
    def rmv_overlaps(keywords):
      ent = [kw[0] for kw in keywords]
      pos = [kw[1] for kw in keywords]
      updated_keywords = []
      n = 0
      for i in range(len(ent)):
        for j in range(len(pos[i])):
          overlap = False
          for k in range(len(ent)):
            for l in range(len(pos[k])):
              if ((pos[i][j][0] >= pos[k][l][0]) and (pos[i][j][1] <= pos[k][l][1]) and i!=k):
                overlap = True
          if (not overlap):
            updated_keywords.append((ent[i],pos[i][j][0],pos[i][j][1]))
      return updated_keywords