from utils.data_structures import Relation
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
        if (type(source)==str and type(target)==str):
            source = [source]
            target = [target]
        if (len(source) != len(target)):
            raise TypeError('The same number of source and target mentions must be provided')
        else:
            self.relations = [Relation(source[i],target[i],self.rel_extractor.compute_relation(source[i], target[i])) for i in range(len(source))]