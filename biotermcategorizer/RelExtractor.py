from itertools import product
from utils.data_structures import Relation
from utils.data_structures import Keyword
from relators.SetFitRelator import SetFitRelator
from relators.TransformersRelator import TransformersRelator

class RelExtractor:
    def __init__(self, 
                 relation_method='transformers', 
                 language='spanish',
                 n=1,
                 thr_transformers=-1,
                 thr_setfit=0.5,
                 all_combinations=False,
                 model_path=None,
                ):
        self.relation_method=relation_method
        self.all_combinations = all_combinations
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
        print(type(source), source)
        if (type(source)==Keyword):
            source = [source]
            print(type(source), source)
        elif (type(source)==str):
            source = [Keyword(text=source)]
            print(type(source), source)
        elif (type(source)==list):
            if (all(isinstance(element, str) for element in source)):
                source = [Keyword(text=i) for i in source]
                print(type(source), source)
            elif not (all(isinstance(element, Keyword) for element in source)):
                print(type(source), source)
                raise TypeError('Source contains elements other than strings or Keyword class objects')
        else:
            raise TypeError('Source must be a string, a Keyword class object, a list of strings or a list of Keyword class objects')
        print(type(source), source)

        print(type(target), target)
        if (type(target)==Keyword):
            target = [target]
            print(type(target), target)
        elif (type(target)==str):
            target = [Keyword(text=target)]
            print(type(target), target)
        elif (type(target)==list):
            if (all(isinstance(element, str) for element in target)):
                target = [Keyword(text=i) for i in target]
                print(type(target), target)
            elif not (all(isinstance(element, Keyword) for element in target)):
                print(type(target), target)
                raise TypeError('Target contains elements other than strings or Keyword class objects')
        else:
            raise TypeError('Target must be a string, a Keyword class object, a list of strings or a list of Keyword class objects')
        print(type(target), target)

        if self.all_combinations:
            source_rep = [i for i,j in list(product(source, target))]
            target_rep = [j for i,j in list(product(source, target))]
            print(type(source_rep), source_rep)
            print(type(target_rep), target_rep)
            relations = self.rel_extractor.compute_relation(source_rep, target_rep)
            self.relations = [Relation(source_rep[i],target_rep[i],relations[i], self.relation_method) for i in range(len(source_rep))]
        else:
            if (len(source) == len(target)):
                relations = self.rel_extractor.compute_relation(source, target)
                self.relations = [Relation(source[i],target[i],relations[i], self.relation_method) for i in range(len(source))]
            else:
                raise TypeError('Source and target must be the same length when all_combinations=False.')