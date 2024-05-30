import pandas as pd
import time
import sys, os, re

# This script can be used to test the TermExtractor pipeline using any desired extraction and categorization method on the given sample files
# Note that the script also returns the computed execution time and saves the results

# Arguments:
# 1 - text files folder path: '../../data/text_files'
# 2 - extraction_method: 'textrank'
# 3 - categorization_method: 'setfit'
# 5 - output_dir path: "./extracted_keywords.tsv"

#set the path to the library
general_path = os.getcwd().split("KeyCARE")[0]+"KeyCARE/"
sys.path.append(general_path+'src/keycare/')
from ..TermExtractor import TermExtractor

start_time = time.time()

keywords = []
extractor = TermExtractor(extraction_method=[sys.argv[2]], categorization_method=sys.argv[3])

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