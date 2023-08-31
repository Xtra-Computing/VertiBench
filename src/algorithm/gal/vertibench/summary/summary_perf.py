import os
import sys
import argparse
import re

import pathlib


def get_score_splitnn(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]

        # The line looks like this
        # Epoch: 49, Test Loss: 1.3684980869293213, Test Score: 0.6394843506622032
        # extract "Test Score" from the line
        score = float(re.findall(r'Test Score: (.+)', last_line)[0])
    return score


def get_scores_splitnn(out_dir, dataset, split='imp', ws=None, bs=None, seed=0):
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

    scores = []
    for ratio in ratios:
        if split == 'imp':
            file_path = os.path.join(out_dir, f"{dataset}_{split}_w{ratio:.1f}_seed{seed}.txt")
        elif split == 'corr':
            # due to a bug, the file name is temporarily w{} instead of b{}
            file_path = os.path.join(out_dir, f"{dataset}_{split}_w{ratio:.1f}_seed{seed}.txt")
        else:
            raise NotImplementedError(f"Splitter {split} is not implemented. "
                                      f"Splitter should be in ['imp', 'corr']")
        score = get_score_splitnn(file_path)
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
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = pathlib.Path(args.log_dir).name

    scores = get_scores_splitnn(args.log_dir, args.dataset,
                                split=args.split, ws=args.weights, bs=args.beta, seed=args.seed)
    for s in scores:
        print(f"{s:.4f}\t", end='')
