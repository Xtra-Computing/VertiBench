import argparse
import os
import sys
import pathlib
from datetime import datetime

from fedtree import FLRegressor, FLClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.VFLDataset import VFLAlignedDataset
from utils import PartyPath


def get_conf(train_dataset_paths, test_dataset_paths, model_path, n_parties, objective, n_classes,
             n_trees, depth, learning_rate, data_format, metric='default') -> str:
    """
    Update the conf file for FedTree
    WARNING: Do not change the *indentations* of the returned string which would break the conf file.
    """

    return f"""
data={','.join(train_dataset_paths)}
test_data={','.join(test_dataset_paths)}
model_path={model_path}
partition_mode=vertical
n_parties={n_parties}
mode=vertical
privacy_tech=none
n_trees={n_trees}
depth={depth}
learning_rate={learning_rate}
partition=0
num_class={n_classes}
objective={objective}
data_format={data_format}
metric={metric}
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters for dataset
    parser.add_argument('--root', '-r', type=str, default='data/syn/')
    parser.add_argument('--dataset', '-d', type=str, default='covtype',
                        help="dataset to use.")
    parser.add_argument('--data-format', '-df', type=str, default='csv',
                        help="data format. Should be in ['csv', 'libsvm']")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    parser.add_argument('--primary_party', '-pp', type=int, default=0,
                        help="primary party. Should be in [0, n_parties-1]")
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")

    # parameters for model
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--lr', '-lr', type=float, default=0.1)
    parser.add_argument('--n_classes', '-c', type=int, default=7,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                             ">=3 for multi-class classification")
    parser.add_argument('--max-depth', '-md', type=int, default=6, help="maximum depth of the tree")
    parser.add_argument('--is-real', '-rd', action='store_true', default=False,
                        help="use real dataset. If not set, use synthetic dataset")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")

    args = parser.parse_args()

    # arg.root was relative to the current working directory, but the binary requires a path relative to the binary
    root_path = os.path.join(args.root, args.dataset)

    # load dataset (prepare paths)
    train_dataset_paths = []
    test_dataset_paths = []
    for i in range(args.n_parties):
        if args.is_real:
            train_dataset_paths.append(os.path.join(args.root, f"{args.dataset}_party{i}_train.csv"))
            test_dataset_paths.append(os.path.join(args.root, f"{args.dataset}_party{i}_test.csv"))
        else:
            data_path_base = os.path.join(root_path, args.dataset)
            train_dataset_paths.append(PartyPath(data_path_base, n_parties=args.n_parties, party_id=i,
                                           splitter=args.splitter, weight=args.weights, beta=args.beta, seed=args.seed,
                                           fmt='csv').data('train'))
            test_dataset_paths.append(PartyPath(data_path_base, n_parties=args.n_parties, party_id=i,
                                            splitter=args.splitter, weight=args.weights, beta=args.beta, seed=args.seed,
                                            fmt='csv').data('test'))

    # get objective according to task
    if args.n_classes == 1:
        objective = 'reg:linear'
        n_class = 1
        metric = 'rmse'
    elif args.n_classes == 2:
        objective = 'binary:logistic'
        metric = 'error'
        n_class = 2
    else:
        objective = 'multi:softmax'
        metric = 'default'
        n_class = args.n_classes

    # define model cache
    cache_path = "cache"
    os.makedirs(cache_path, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(cache_path, f"fedtree_{time_stamp}.model")

    # save conf file
    conf_content = get_conf(train_dataset_paths, test_dataset_paths, model_path=model_path, n_parties=args.n_parties,
                            objective=objective, n_classes=n_class, n_trees=args.epochs, depth=args.max_depth,
                            learning_rate=args.lr, data_format=args.data_format, metric=metric)
    os.makedirs("algo/FedTree/conf", exist_ok=True)
    if args.is_real:
        conf_path = f"algo/FedTree/conf/fedtree-vertical-{args.dataset}-{args.n_parties}-{args.seed}.conf"
    else:
        ratio = args.weights if args.splitter == 'imp' else args.beta
        conf_path = f"algo/FedTree/conf/fedtree-vertical-{args.dataset}-{args.n_parties}-{args.splitter}-{ratio}-{args.seed}.conf"
    pathlib.Path(conf_path).write_text(conf_content)

    # train model by invoking the FedTree binary
    binary_path = "./algo/FedTree/build/bin/FedTree-train"
    if not os.path.exists(binary_path):
        raise FileNotFoundError("FedTree binary not found. Please build the binary first.")
    os.system(f"{binary_path} {conf_path}")
