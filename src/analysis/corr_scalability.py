import os
import re
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def split_syn_data():
    root_dir = "data/syn/sklearn"
    # for n_samples in [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
    for n_samples in [5000000, 10000000]:
        n_features = 100
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=n_features//2,
                                   n_repeated=0, n_classes=2, random_state=0, shuffle=False)

        save_path = os.path.join(root_dir, f"syn_{n_samples}_{n_features}.csv")
        pd.DataFrame(X).to_csv(save_path, index=False)
        print(f"Saved to {save_path}")

    # # generate dataset of different dimensions, save to data/syn/sklearn
    # os.makedirs(root_dir, exist_ok=True)
    # for n_features in [10, 100, 500, 1000, 5000, 10000, 50000, 100000]:
    #     n_samples = 1000
    #     X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=n_features//2,
    #                                n_repeated=0, n_classes=2, random_state=0, shuffle=False)
    #
    #     save_path = os.path.join(root_dir, f"syn_{n_samples}_{n_features}.csv")
    #     pd.DataFrame(X).to_csv(save_path, index=False)
    #     print(f"Saved to {save_path}")


def summary_time(out_dir="out/time/"):
    n_features_list = [10, 100, 500, 1000, 5000, 10000]
    n_samples_list = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

    plt.rcParams.update({'font.size': 16})

    i_times = []
    for n_samples in n_samples_list:
        n_features = 100
        path = os.path.join(out_dir, f"syn_{n_samples}_{n_features}.txt")
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Time cost" in line:
                    time_cost = float(re.findall(r"Time cost: (\d+\.\d+)s", line)[0])
                    i_times.append(time_cost)
                    print(f"{n_samples=}, {n_features=}, {time_cost=}")
                    break

    # plot time cost for different samples
    plt.figure()
    plt.plot(n_samples_list, i_times, label='Split time')
    plt.xscale('log')
    plt.xlabel('Number of samples')
    plt.ylabel('Time cost (sec)')
    plt.tight_layout()
    plt.savefig("fig/syn_time_samples.png")

    f_times = []
    for n_features in n_features_list:
        n_samples = 1000
        path = os.path.join(out_dir, f"syn_{n_samples}_{n_features}.txt")
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Time cost" in line:
                    time_cost = float(re.findall(r"Time cost: (\d+\.\d+)s", line)[0])
                    f_times.append(time_cost)
                    print(f"{n_samples=}, {n_features=}, {time_cost=}")
                    break

    # plot time cost for different dimensions
    plt.figure()
    plt.plot(n_features_list, f_times, label='Split time')
    plt.xscale('log')
    plt.xlabel('Number of features')
    plt.ylabel('Time cost (sec)')
    plt.tight_layout()
    plt.savefig("fig/syn_time_features.png")

def summary_time_party():
    plt.rcParams.update({'font.size': 16})

    n_instances = 1000
    n_features = 1000
    # 2 4 10 25 100 400 800 1000
    n_parties_list = [2, 4, 10, 25, 100, 400, 800, 1000]

    times = []
    for n_parties in n_parties_list:
        path = f"out/time/syn_{n_instances}_{n_features}_{n_parties}.txt"
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Time cost" in line:
                    time_cost = float(re.findall(r"Time cost: (\d+\.\d+)s", line)[0])
                    times.append(time_cost)
                    print(f"{n_instances=}, {n_features=}, {n_parties=}, {time_cost=}")
                    break

    # plot time cost for different dimensions
    plt.figure()
    plt.plot(n_parties_list, times, label='Split time')
    plt.xscale('log')
    plt.ylim((1000, 1600))
    plt.xlabel('Number of parties')
    plt.ylabel('Time cost (sec)')
    plt.tight_layout()
    plt.savefig("fig/syn_time_parties.png")



if __name__ == '__main__':
    # split_syn_data()
    # summary_time()
    summary_time_party()