#Test de extracción completa directa
text = "Un varón de 32 años acude al Servicio de Urgencias por disminución reciente de visión en OD coincidiendo con la aparición de una lesión parduzca en dicho ojo. Entre los antecedentes oftalmológicos destaca un traumatismo penetrante en OD tres años antes que fue suturado en nuestro centro. A la exploración presenta una agudeza visual de 0,1 que mejora a 0,5 con estenopéico."

extractor1 = TermExtractor(extraction_methods="textrank", language="spanish", max_tokens=5)
print("Usando TextRank y un máximo de 5 tokens: \n",extractor1(text))

extractor2 = TermExtractor(extraction_methods=["rake", "yake"], language="spanish", max_tokens=2)
print("\nUsando Rake y Yake juntos y un máximo de 2 tokens: \n",extractor2(text))

extractor3 = TermExtractor(extraction_methods=["rake", "yake", "textrank"], language="spanish", max_tokens=3)
print("\nUsando Rake, Yake y TextRank juntos y un máximo de 3 tokens: \n",extractor3(text))

#Test de extracción por pasos: vemos que da lo mismo que el de antes
extractor = TermExtractor(extraction_methods="textrank", language="spanish", max_tokens=3)
terms = extractor.extractors["textrank"].extract_terms(text)
terms_with_span = extractor.extractors["textrank"].find_term_span(text,terms)
terms_without_overlaps = extractor.extractors["textrank"].rmv_overlaps(terms_with_span)
print("Usando TextRank y un máximo de 5 tokens paso por paso: \n",terms_without_overlaps)