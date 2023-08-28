from utils import PartyPath
from FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from dataset import VFLAlignedDataset


def evaluate_split(datadir, dataset, n_parties, splitter, weight, beta, seed, data_type):
    data = VFLAlignedDataset.from_pickle(datadir, dataset, n_parties, 0, splitter, weight, beta, seed, data_type)
    if splitter == 'imp':
        evaluator = ImportanceEvaluator()
    elif splitter == 'corr':
        evaluator = CorrelationEvaluator()
    else:
        raise NotImplementedError(f"Splitter {splitter} is not implemented. Splitter should be in ['imp', 'corr']")

    evaluator.evaluate(data)