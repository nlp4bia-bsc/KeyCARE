import pandas as pd
import time
import sys, os, re

# Arguments:
# 1 - text files folder path: '/mnt/c/Users/Sergi/Desktop/BSC/Corpus/distemist_zenodo/training/text_files2'
# 2 - extraction_method: 'textrank'
# 3 - categorization_method: 'setfit'
# 5 - output_dir path: "/mnt/c/Users/Sergi/Desktop/BSC/extracted_keywords.tsv"

#set the path to the library
general_path = os.getcwd().split("BioTermCategorizer")[0]+"BioTermCategorizer/"
sys.path.append(general_path+'biotermcategorizer/')
from TermExtractor import TermExtractor

start_time = time.time()

keywords = []
extractor = TermExtractor(extraction_method=[sys.argv[2]], categorization_method=sys.argv[3]) #parametros como args de entrada

docs = os.listdir(sys.argv[1])
for doc in docs:
    if doc.endswith(".txt"):
        with open(os.path.join(sys.argv[1], doc), "r") as file:
            text = file.read()
        extractor(text)
        kws = extractor.keywords
        kws_with_file = [(doc[:-4], kw.span, kw.text, kw.score, kw.label, kw.extraction_method, kw.categorization_method) for kw in kws]
        keywords.extend(kws_with_file)

df = pd.DataFrame(keywords, columns=["file", "span", "text", "score", "label", "extraction_method", "categorization_method"])
df.to_csv(sys.argv[4], sep="\t")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")