import os
import sys

from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from preprocess.FeatureEvaluator import ImportanceEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter

plt.rcParams['font.size'] = 16


def plot_alpha_vs_std(feature_importance):
    # split with different consistent alpha, estimate variance of feature importance
    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    party_imp_stds = []
    for alpha in alphas:
        feature_splitter = ImportanceSplitter(4, alpha)
        party_imps = []
        for i in range(1000):
            feature_imp_per_party = feature_splitter.split(feature_importance.reshape(1, -1), allow_empty_party=True)
            party_imp = [np.sum(feature_imp) for feature_imp in feature_imp_per_party]
            party_imps.append(party_imp)
        party_imps = np.array(party_imps)
        party_imp_mean = np.mean(party_imps, axis=0)
        party_imp_mean_std = np.mean(np.std(party_imps, axis=1))
        print(f"alpha: {alpha}, party_imp_mean: {party_imp_mean}, party_imp_mean_std: {party_imp_mean_std}")
        party_imp_stds.append(party_imp_mean_std)
    uniform_std = party_imp_stds[2]     # std of uniform distribution (alpha=1)

    # plot alpha vs party_imp_mean_std in a figure
    fig, ax = plt.subplots()
    ax.plot(alphas, party_imp_stds, marker='o')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("Averaged standard deviation")
    ax.set_xscale("log")

    # plot uniform distribution in -- and annotate in the left middle above the line
    ax.plot([0.01, 1000], [uniform_std, uniform_std], linestyle='--', color='red')
    ax.annotate("Random Split", xy=(0.1, uniform_std), xytext=(3, uniform_std + 1), color='red')

    # ax.set_title("Averaged standard deviation of party importance under different $\\alpha$ of Dirichlet split")
    fig.savefig("fig/alpha_vs_std.png", bbox_inches='tight')

def plot_alpha_vs_mean_imp(feature_importance):
    alphas = [0.01, 0.1, 0.5, 1, 2, 4, 8, 10, 50, 100, 1000]
    alpha_others = 1

    party_mean_imps = []
    scaled_alphas = []
    for alpha in alphas:
        alpha_all = [alpha] + [alpha_others]
        scaled_alphas.append(np.array(alpha_all) / np.sum(alpha_all))
        feature_splitter = ImportanceSplitter(2, alpha_all)
        party_imps = []
        for i in range(1000):
            feature_imp_per_party = feature_splitter.split(feature_importance.reshape(1, -1), allow_empty_party=True)
            party_imp = [np.sum(feature_imp) for feature_imp in feature_imp_per_party]
            # scaled_party_imp = np.array(party_imp) / np.sum(party_imp)
            party_imps.append(party_imp)
        party_imps = np.array(party_imps)
        party_imp_mean = np.mean(party_imps, axis=0)
        party_mean_imps.append(party_imp_mean)
    party_mean_imps = np.array(party_mean_imps)
    scaled_alphas = np.array(scaled_alphas)

    # plot alpha vs party_first_imp in a figure
    fig, ax = plt.subplots()
    ax.plot(scaled_alphas[:, 0], party_mean_imps[:, 0], marker='o', label="Party $P_0$", color='blue')

    # two x-axis: top - party 1 (reverse), bottom - party 0
    ax2 = ax.twiny()
    ax2.plot(scaled_alphas[:, 1], party_mean_imps[:, 1], marker='^', label="Party $P_1$", color='orange')
    ax2.invert_xaxis()
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax2.set_xlabel("Scaled $\\alpha$ of $P_1$")

    ax.set_xlabel("Scaled $\\alpha$ of $P_0$")
    ax.set_ylabel("Averaged party importance")

    # combined legend of two x-axis
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='right')

    # ax.set_title("Averaged first party importance under different $\\alpha$ of Dirichlet split")
    fig.savefig("fig/alpha_vs_mean_imp.png", bbox_inches='tight')



if __name__ == '__main__':
    # load data
    X, y = load_svmlight_file("data/syn/letter/letter.libsvm")
    X = X.toarray()
    print(X.shape, y.shape)

    # train a model on the data
    model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=50, max_depth=6, learning_rate=0.1,
                              objective='multi:softmax', num_class=26)
    model.fit(X, y)
    print("Model trained.")

    evaluator = ImportanceEvaluator(sample_rate=0.001)
    feature_importance = evaluator.evaluate_feature(X, model.predict)
    print(feature_importance)

    plot_alpha_vs_mean_imp(feature_importance)
    plot_alpha_vs_std(feature_importance)









