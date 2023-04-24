import os
import sys
import argparse
import re

import pathlib


def get_log_paths(out_dir, dataset, split='imp', ws=None, bs=None, seed=0):
    if ws is None:
        ws = [0.1, 0.3, 0.6, 1.0]
    if bs is None:
        bs = [0.0, 0.3, 0.6, 1.0]

    if split == 'imp':
        ratios = ws
    elif split == 'corr':
        ratios = bs
    else:
        raise NotImplementedError(f"Splitter {split} is not implemented. "
                                  f"Splitter should be in ['imp', 'corr']")

    log_paths = []
    for ratio in ratios:
        if split == 'imp':
            file_path = os.path.join(out_dir, dataset, f"{dataset}_{split}_w{ratio:.1f}_seed{seed}.txt")
        elif split == 'corr':
            # due to a bug, the file name is temporarily w{} instead of b{}
            file_path = os.path.join(out_dir, dataset, f"{dataset}_{split}_w{ratio:.1f}_seed{seed}.txt")
        else:
            raise NotImplementedError(f"Splitter {split} is not implemented. "
                                      f"Splitter should be in ['imp', 'corr']")
        log_paths.append(file_path)
    return log_paths


def get_score_splitnn(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]

        # The line looks like this
        # Epoch: 49, Test Loss: 1.3684980869293213, Test Score: 0.6394843506622032
        # extract "Test Score" from the line
        score = float(re.findall(r'Test Score: (.+)', last_line)[0])
    return score


def get_score_fedtree(file_path):
    """
    extract the last match of score from the log file.
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # The output fragment that contains score looks like "INFO gbdt.cpp:182 : AUC = 0.996197"
        # We need to extract 0.996197 from the line. The score is a float value or int: 1, 1.0, 0.665, etc.
        # The metric may not be AUC, but it should be one or multiple words.
        # The line number may not be 182, but it should be a single number.
        # The line may not be the last line, but it should be the last line that contains the score.
        score = None
        for line in reversed(lines):
            match = re.findall(r'INFO gbdt.cpp:(\d+) : (.+) = (\d+\.?\d*)', line)
            if match:
                score = float(match[0][2])
                break
    return score

def get_scores_splitnn(out_dir, dataset, split='imp', ws=None, bs=None, seed=0):
    log_paths = get_log_paths(out_dir, dataset, split=split, ws=ws, bs=bs, seed=seed)
    scores = []
    for log_path in log_paths:
        score = get_score_splitnn(log_path)
        scores.append(score)
    return scores

def get_scores_fedtree(out_dir, dataset, split='imp', ws=None, bs=None, seed=0):
    log_paths = get_log_paths(out_dir, dataset, split=split, ws=ws, bs=bs, seed=seed)
    scores = []
    for log_path in log_paths:
        score = get_score_fedtree(log_path)
        scores.append(score)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str)
    parser.add_argument('--dataset', '-d', type=str, default=None)
    parser.add_argument('--split', '-sp', type=str, default='imp', help="splitter type, should be in ['imp', 'corr']")
    parser.add_argument('--weights', '-w', type=list, default=None, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=list, default=None, help="beta for the CorrelationSplitter")
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--algo', '-a', type=str, default='splitnn',
                        help="algorithm, should be in ['splitnn', 'fedtree']")
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = pathlib.Path(args.log_dir).name

    if args.algo == 'splitnn':
        scores = get_scores_splitnn(args.log_dir, args.dataset,
                                    split=args.split, ws=args.weights, bs=args.beta, seed=args.seed)
    elif args.algo == 'fedtree':
        scores = get_scores_fedtree(args.log_dir, args.dataset,
                                    split=args.split, ws=args.weights, bs=args.beta, seed=args.seed)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented. "
                                  f"Algorithm should be in ['splitnn', 'fedtree']")
    for s in scores:
        print(f"{s:.4f}\t", end='')
