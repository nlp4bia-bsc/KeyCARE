from utils.data_structures import Relation
from utils.data_structures import Keyword
from relators.SetFitRelator import SetFitRelator
from relators.TransformersRelator import TransformersRelator

class RelExtractor:
    def __init__(self, 
                 relation_method='transformers', 
                 language='spanish',
                 n=1,
                 thr_transformers=0,
                 thr_setfit=0.5,
                 model_path=None,
                ):
        self.relation_method=relation_method
        self.rel_extractor = self.initialize_relation_method(language, n, thr_transformers, thr_setfit, model_path)
        
    def initialize_relation_method(self, language, n, thr_transformers, thr_setfit, model_path):
        if 'transformers' == self.relation_method:
            rel_extractor = TransformersRelator(n, thr_transformers, model_path)
        elif 'setfit' == self.relation_method:
            rel_extractor = SetFitRelator(n, thr_setfit, model_path)
        else:
            raise ValueError("No relation method called {}".format(self.relation_method))
        return rel_extractor
    
    def __call__(self, source, target):
        if (type(source)==Keyword and type(target)==Keyword):
            source = [source]
            target = [target]
        elif (type(source)==str and type(target)==str):
            source = [Keyword(text=source)]
            target = [Keyword(text=target)]
        elif (type(source)==list and type(target)==list):
            if (all(isinstance(element, str) for element in source) and all(isinstance(element, str) for element in target)):
                source = [Keyword(text=i) for i in source]
                target = [Keyword(text=j) for j in target]
            elif not (all(isinstance(element, Keyword) for element in source) and all(isinstance(element, Keyword) for element in target)):
                raise TypeError('Source and target contain elements other than strings or Keyword class objects')
        else:
            raise TypeError('Source and target must be strings, Keyword class objects, lists of strings or lists of Keyword class objects')
        if (len(source) != len(target)):
            raise TypeError('The same number of source and target mentions must be provided')
        else:
            self.relations = [Relation(source[i],target[i],self.rel_extractor.compute_relation(source[i].text, target[i].text)) for i in range(len(source))]