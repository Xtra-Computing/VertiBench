import time
import gc
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import CorrelationEvaluator
from utils.utils import PartyPath
from dataset.VFLDataset import VFLSynAlignedDataset


def plot_dt_acc_and_time(dataset, n_points=10, process='mcor', n_iter=4):
    # plot the relationship between n_components (d_t) of truncated SVD vs. accuracy and time

    # load data
    vfl_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", dataset, 4, 0, 'imp', 1)
    Xs = vfl_dataset.Xs
    n_features = sum([X.shape[1] for X in Xs])
    print(f"{n_features=}")

    if isinstance(n_points, int):
        n_components_list = np.arange(1, n_features + n_features // n_points, n_features // n_points)
    else:
        n_components_list = n_points

    speedup_list = []
    error_list = []
    speedup_std_list = []
    error_std_list = []

    exact_evaluator = CorrelationEvaluator(gpu_id=0, svd_algo='exact')
    exact_evaluator.fit(Xs)

    approx_evaluator = CorrelationEvaluator(gpu_id=0, svd_algo='approx')
    approx_evaluator.fit(Xs)

    for n_components in n_components_list:
        # estimate correlation
        start_exact = time.time()
        if process == 'icor':
            exact_score = exact_evaluator.evaluate()
        elif process == 'mcor':
            exact_score = exact_evaluator.mcor_singular_exact_gpu(exact_evaluator.corr)
        else:
            assert False
        end_exact = time.time()

        exact_duration_sec = end_exact - start_exact
        print(f"{exact_score=}, {exact_duration_sec=}")

        error_summary = []
        speedup_summary = []
        for i in range(1):
            start_approx = time.time()
            if process == 'icor':
                approx_score = approx_evaluator.evaluate()
            elif process == 'mcor':
                approx_score = approx_evaluator.mcor_singular_approx_gpu(approx_evaluator.corr,
                                                                         n_components=n_components, n_iter=n_iter)
            else:
                assert False
            end_approx = time.time()

            approx_duration_sec = end_approx - start_approx

            error = np.abs(exact_score - approx_score) / exact_score
            error_summary.append(error)
            speedup = exact_duration_sec / approx_duration_sec
            speedup_summary.append(speedup)
        mean_error = np.mean(error_summary)
        mean_speedup = np.mean(speedup_summary)
        std_error = np.std(error_summary)
        std_speedup = np.std(speedup_summary)
        print(f"{n_components=}, {mean_error=}, {mean_speedup=}, {std_error=}, {std_speedup=}")

        error_list.append(mean_error)
        speedup_list.append(mean_speedup)
        error_std_list.append(std_error)
        speedup_std_list.append(std_speedup)

    # plot, two y-axis, one for speedup, one for error, x-axis for n_components
    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("$d_t$")
    ax1.set_ylabel('Speedup')
    # ax1.errorbar(n_components_list, speedup_list, yerr=speedup_std_list, marker='o', label='Speedup', color='blue',
    #              markersize=5)
    ax1.plot(n_components_list, speedup_list, marker='o', label='Speedup', color='blue',
                 markersize=5)
    ax1.set_ylim([0, None])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Error')
    # ax2.errorbar(n_components_list, error_list, yerr=error_std_list, marker='^', label='Error', color='red',
    #              markersize=5)
    ax2.plot(n_components_list, error_list, marker='^', label='Error', color='red',
                    markersize=5)
    if dataset in ['epsilon']:
        ax2.set_ylim([0, 0.1])
    elif dataset in ['realsim']:
        ax2.set_ylim([0, 1])
    else:
        ax2.set_ylim([0, 0.5])

    # x-axis log scale
    if dataset not in ['letter', 'covtype']:
        ax1.set_xscale('log')
        ax2.set_xscale('log')

    # combined legend of two y-axis
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.title(f"{dataset}")

    fig.tight_layout()
    fig.savefig(f"fig/svd-dt-{dataset}.png")


if __name__ == '__main__':
    datasets = ['covtype', 'letter', 'radar', 'msd', 'gisette', 'epsilon', 'realsim']
    n_features_list = [54, 16, 174, 90, 5000, 2000, 72309, 20958]

    plot_dt_acc_and_time('letter', [10, 11, 12, 13, 14, 15, 16])
    plot_dt_acc_and_time('covtype', [1, 2, 3, 4, 5, 6, 7])
    plot_dt_acc_and_time('radar', [4, 6, 8, 16, 32, 64, 128, 174])
    plot_dt_acc_and_time('msd', [8, 16, 32, 64, 72, 84, 90])
    plot_dt_acc_and_time('gisette', [1, 2, 4, 8, 16, 64, 128, 256, 384, 512])
    plot_dt_acc_and_time('realsim', [1, 4, 16, 64, 128, 256, 384, 512])
    plot_dt_acc_and_time('epsilon', [1, 4, 16, 64, 256, 512, 768, 1024, 2000])






