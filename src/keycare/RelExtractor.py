from itertools import product
from .utils.data_structures import Relation
from .utils.data_structures import Keyword
from .relators.SetFitRelator import SetFitRelator
from .relators.TransformersRelator import TransformersRelator

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
        """
        Initializes the RelExtractor class.

        Parameters:
        relation_method (str): Method used for relation extraction.
        language (str): Language for text processing.
        n (int): Maximum number of labels for a single relation.
        thr_setfit (float): Threshold for SetFit relation classification.
        thr_transformers (float): Threshold for Transformers relation classification.
        all_combinations (bool): Whether to create relations for all combinations of source and target.
        model_path (str): Path to the model if required.

        """
        self.relation_method=relation_method
        self.all_combinations = all_combinations
        self.rel_extractor = self.initialize_relation_method(language, n, thr_transformers, thr_setfit, model_path)
        
    def initialize_relation_method(self, language, n, thr_transformers, thr_setfit, model_path):
        """
        Initializes the selected relation extraction method.

        Parameters:
        language (str): Language for text processing.
        n (int): Maximum number of labels for a single relation.
        thr_setfit (float): Threshold for SetFit relation classification.
        thr_transformers (float): Threshold for Transformers relation classification.
        model_path (str): Path to the model if required.

        Returns:
        object: Instance of the selected relation extraction method.
        """
        if 'transformers' == self.relation_method:
            rel_extractor = TransformersRelator(n, thr_transformers, model_path)
        elif 'setfit' == self.relation_method:
            rel_extractor = SetFitRelator(n, thr_setfit, model_path)
        else:
            raise ValueError("No relation method called {}".format(self.relation_method))
        return rel_extractor
    
    def __call__(self, source, target):
        """
        Executes the relation extraction based on the source and target inputs.

        Parameters:
        source (str/Keyword/list): Source for relation extraction.
        target (str/Keyword/list): Target for relation extraction.

        Raises:
        TypeError: If source or target are not of expected types or lengths.

        Returns:
        list: List of Relation objects based on the extraction.
        """
        if (type(source)==Keyword):
            source = [source]
        elif (type(source)==str):
            source = [Keyword(text=source)]
        elif (type(source)==list):
            if (all(isinstance(element, str) for element in source)):
                source = [Keyword(text=i) for i in source]
            elif not (all(isinstance(element, Keyword) for element in source)):
                raise TypeError('Source contains elements other than strings or Keyword class objects')
        else:
            raise TypeError('Source must be a string, a Keyword class object, a list of strings or a list of Keyword class objects')

        if (type(target)==Keyword):
            target = [target]
        elif (type(target)==str):
            target = [Keyword(text=target)]
        elif (type(target)==list):
            if (all(isinstance(element, str) for element in target)):
                target = [Keyword(text=i) for i in target]
            elif not (all(isinstance(element, Keyword) for element in target)):
                raise TypeError('Target contains elements other than strings or Keyword class objects')
        else:
            raise TypeError('Target must be a string, a Keyword class object, a list of strings or a list of Keyword class objects')

        if self.all_combinations:
            source_rep = [i for i,j in list(product(source, target))]
            target_rep = [j for i,j in list(product(source, target))]
            relations = self.rel_extractor.compute_relation(source_rep, target_rep)
            self.relations = [Relation(source_rep[i],target_rep[i],relations[i], self.relation_method) for i in range(len(source_rep))]
        else:
            if (len(source) == len(target)):
                relations = self.rel_extractor.compute_relation(source, target)
                self.relations = [Relation(source[i],target[i],relations[i], self.relation_method) for i in range(len(source))]
            else:
                raise TypeError('Source and target must be the same length when all_combinations=False.')