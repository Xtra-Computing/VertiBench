import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_classification, load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from preprocess.FeatureEvaluator import CorrelationEvaluator, ImportanceEvaluator
from preprocess.FeatureSplitter import CorrelationSplitter, ImportanceSplitter


# matplotlib.use('Agg')


def test_corr_splitter():
    # Generate a synthetic dataset using sklearn
    # X1, y1 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=0, shuffle=True)
    # X2, y2 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=1, shuffle=True)
    # X3, y3 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=2, shuffle=True)
    # X = np.concatenate((X1, X2, X3), axis=1)
    # X = load_svmlight_file("data/syn/gisette_scale")[0].toarray()
    X = load_svmlight_file("data/syn/covtype/covtype")[0].toarray()
    # scale each feature to [0, 1]
    X = MinMaxScaler().fit_transform(X)
    # Remove constant features
    # X = X[:, np.std(X, axis=0) > 0]
    print(X.shape)
    corr = spearmanr(X).correlation     # this correlation is only used for plotting
    corr = np.nan_to_num(corr, nan=0)
    corr_evaluator = CorrelationEvaluator()
    corr_spliter = CorrelationSplitter(num_parties=3, evaluator=corr_evaluator)

    corr_spliter.fit(X, verbose=True)
    print(f"Min mcor: {corr_spliter.min_mcor}, Max mcor: {corr_spliter.max_mcor}")
    for beta in [0, 0.33, 0.66, 1]:
        Xs = corr_spliter.split(X, beta=beta, verbose=True)
        eval_mcor = corr_evaluator.evaluate(Xs)

        corr_perm = corr[corr_spliter.best_permutation, :][:, corr_spliter.best_permutation]
        print(f"{beta=}: best_mcor={corr_spliter.best_mcor:.4f}, eval_mcor={eval_mcor:.4f}, "
              f"best_error={corr_spliter.best_error:.6f}, features_per_party={corr_spliter.best_feature_per_party}")

        # plot permuted correlation matrix in heatmap
        plt.figure()
        # add sigmoid to make the color more distinguishable
        # corr_perm = np.exp(corr_perm) - 1
        plt.imshow(corr_perm, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'beta={beta}, best_mcor={corr_spliter.best_mcor:.4f}')
        plt.show()


def test_importance_splitter_diff_alpha(n_rounds=100):

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                               n_classes=2, random_state=0, shuffle=True)
    X = MinMaxScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)

    # different weights for each party
    importance_per_w = []
    importance_per_w_std = []
    w_range = np.arange(1, 4, 0.5)
    for w in w_range:
        importance_summary = []
        for i in tqdm(range(n_rounds)):
            splitter = ImportanceSplitter(num_parties=3, weights=[3, 2, w], seed=i)
            Xs = splitter.split(X)
            evaluator = ImportanceEvaluator(model.predict, sample_rate=0.01, seed=i)
            party_importance = evaluator.evaluate(Xs)
            # print(f"Party importance {party_importance}")
            importance_summary.append(party_importance)
        importance_summary = np.array(importance_summary)
        mean_importance = np.mean(importance_summary, axis=0)
        std_importance = np.std(importance_summary, axis=0)
        print(f"Mean importance: {mean_importance}")
        print(f"Std importance: {std_importance}")
        importance_per_w.append(mean_importance)
        importance_per_w_std.append(std_importance)
    importance_per_w = np.array(importance_per_w).T
    importance_per_w_std = np.array(importance_per_w_std).T
    # plot the importance as w changes
    plt.figure()
    plt.plot(w_range, importance_per_w[0], marker='o')
    plt.plot(w_range, importance_per_w[1], marker='x')
    plt.plot(w_range, importance_per_w[2], marker='^')
    plt.legend([r"Party 1 ($\alpha_1=3$)", r"Party 2 ($\alpha_2=2$)", r"Party 3"])
    plt.xlabel(r"Weight of Party 3 ($\alpha_3$)")
    plt.ylabel("Shapley importance")
    plt.title(r"Shapley importance as $\alpha_3$ changes")
    plt.show()


def test_importance_splitter_same_alpha(n_rounds=100):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                               n_classes=2, random_state=0, shuffle=True)
    X = MinMaxScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)

    # same but varies weights for each party
    mean_importance_per_w = []
    std_importance_per_w = []
    w_range = [0.1, 0.5, 1, 2, 5, 10]
    for w in w_range:
        importance_summary = []
        for i in tqdm(range(n_rounds)):
            splitter = ImportanceSplitter(num_parties=10, weights=w, seed=i)
            Xs = splitter.split(X)
            evaluator = ImportanceEvaluator(model.predict, sample_rate=0.02, seed=i)
            party_importance = evaluator.evaluate(Xs)
            # print(f"Party importance {party_importance}")
            importance_summary.append(party_importance)
        importance_summary = np.array(importance_summary)
        mean_importance = np.mean(importance_summary, axis=0)
        std_importance = np.mean(np.std(importance_summary, axis=1))  # std among parties
        print(f"Mean importance: {mean_importance}")
        print(f"Std importance: {std_importance}")
        mean_importance_per_w.append(mean_importance)
        std_importance_per_w.append(std_importance)
    mean_importance_per_w = np.array(mean_importance_per_w).T
    std_importance_per_w = np.array(std_importance_per_w)
    # plot the mean importance and std as w changes
    plt.figure()
    plt.plot(w_range, std_importance_per_w, marker='o')
    # plt.legend([rf"Party {i} ($\alpha_{i}$=$\alpha$)" for i in range(1, 4)])
    plt.xlabel(r"Weight of each party ($\alpha$)")
    plt.ylabel("Standard variance")
    plt.title(r"Standard variance of Shapley importance across parties")
    plt.show()


def test_weight_different_alpha():
    w_range = np.arange(1, 4, 0.5)
    w_party1_base = 3
    w_party2_base = 2
    scale_ws = []
    for w in w_range:
        w_party1_scale = w_party1_base / (w_party1_base + w_party2_base + w)
        w_party2_scale = w_party2_base / (w_party1_base + w_party2_base + w)
        w_party3_scale = w / (w_party1_base + w_party2_base + w)
        scale_ws.append([w_party1_scale, w_party2_scale, w_party3_scale])
    scale_ws = np.array(scale_ws)
    plt.figure()
    plt.plot(w_range, scale_ws[:, 0], marker='o')
    plt.plot(w_range, scale_ws[:, 1], marker='x')
    plt.plot(w_range, scale_ws[:, 2], marker='^')
    plt.legend([r"Party 1 ($\alpha_1$)", r"Party 2 ($\alpha_2$)", r"Party 3 ($\alpha_3$)"])
    plt.xlabel(r"Weight of Party 3 ($\alpha_3$)")
    plt.ylabel("Scaled weight")
    plt.title(r"Scaled weight of each party as $\alpha_3$ changes")
    plt.show()


def test_eval():
    n_parties = 10
    party_corr0_summary = []
    party_corr1_summary = []
    party_importance_summary = []
    acc_summary = []
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=1, n_estimators=50, max_depth=5, learning_rate=0.1)
    gamma = 1
    for seed in range(10):
        X, y = make_classification(n_samples=10000, n_features=200, n_informative=5, n_redundant=0, n_repeated=195,
                                   n_classes=2, random_state=seed, shuffle=True, class_sep=0.5)
        X = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # print(f"Combine accuracy: {accuracy_score(y_test, y_pred)}")

        # Random split into multiple parties
        splitter = ImportanceSplitter(num_parties=n_parties, weights=[10] + [50] * (n_parties - 1), seed=seed)
        Xs_train, Xs_test = splitter.split(X_train, X_test)

        # continue only when all the X has non-zero dimensions
        if not all([X.shape[1] > 0 for X in Xs_train]):
            continue

        imp_evaluator = ImportanceEvaluator(model.predict, sample_rate=0.01, seed=seed)
        # party_importance = imp_evaluator.evaluate(Xs_train)[1:] # exclude the primary party
        # print(f"Party importance {party_importance}")
        corr_evaluator0 = CorrelationEvaluator(gamma=0)
        corr_evaluator1 = CorrelationEvaluator(gamma=1)
        party_corr0 = []
        party_corr1 = []
        X_primary = Xs_train[0]
        for i in range(1, len(Xs_train)):
            # corr = spearmanr(np.concatenate([X_primary, Xs_train[i]], axis=1)).correlation
            # inter_corr = CorrelationEvaluator.mcor_singular(corr[X_primary.shape[1]:, :X_primary.shape[1]])
            # inner_corr = CorrelationEvaluator.mcor_singular(corr[:X_primary.shape[1], :X_primary.shape[1]])
            # inner_corr2 = CorrelationEvaluator.mcor_singular(corr[X_primary.shape[1]:, X_primary.shape[1]:])
            party_corr0.append(corr_evaluator0.evaluate([X_primary, Xs_train[i]]))
            party_corr1.append(corr_evaluator1.evaluate([X_primary, Xs_train[i]]))
            # party_corr.append(inter_corr - inner_corr)
        print(f"Party correlation gamma=0 {party_corr0}")
        print(f"Party correlation gamma=1 {party_corr1}")

        # test training in each collaboration (between the primary party and each secondary party)
        model.fit(Xs_train[0], y_train)
        acc_scores = []
        party_importance = []
        for i in range(1, len(Xs_train)):
            X_train_i = np.concatenate([Xs_train[0], Xs_train[i]], axis=1)
            X_test_i = np.concatenate([Xs_test[0], Xs_test[i]], axis=1)
            model.fit(X_train_i, y_train)
            y_pred = model.predict(X_test_i)
            acc = accuracy_score(y_test, y_pred)
            acc_scores.append(acc)

            # party_importance.append(imp_evaluator.evaluate([Xs_test[0], Xs_test[i]])[1])
        print(f"Party importance {party_importance}")

        party_corr0_summary.append(party_corr0)
        party_corr1_summary.append(party_corr1)
        party_importance_summary.append(party_importance)
        acc_summary.append(acc_scores)

    party_corr0_summary = np.array(party_corr0_summary).flatten()
    party_corr1_summary = np.array(party_corr1_summary).flatten()
    party_importance_summary = np.array(party_importance_summary).flatten()
    acc_summary = np.array(acc_summary).flatten()
    corr0_acc_spearman = spearmanr(party_corr0_summary, acc_summary).correlation
    corr1_acc_spearman = spearmanr(party_corr1_summary, acc_summary).correlation
    # imp_acc_spearman = spearmanr(party_importance_summary, acc_summary).correlation




    # # plot how party importance and correlation affects the accuracy in two separate line charts
    # plt.figure()
    # plt.scatter(party_importance_summary, acc_summary, marker='o')
    # # add a regression line
    # z = np.polyfit(party_importance_summary, acc_summary, 1)
    # p = np.poly1d(z)
    # plt.plot(party_importance_summary, p(party_importance_summary), "r--")
    # plt.xlabel("Party importance")
    # plt.ylabel("Accuracy")
    # plt.title(f"Party importance vs accuracy (#parties = {n_parties})")
    # plt.show()
    plt.figure()
    corr0_order = np.argsort(party_corr0_summary)
    plt.scatter(party_corr0_summary[corr0_order], acc_summary[corr0_order], marker='o')
    # add a regression line
    z = np.polyfit(party_corr0_summary[corr0_order], acc_summary[corr0_order], 2)
    py = np.polyval(z, party_corr0_summary[corr0_order])
    plt.plot(party_corr0_summary[corr0_order], py, "r--")
    plt.xlabel("Party correlation")
    plt.ylabel("Accuracy")
    plt.title(f"Party correlation vs accuracy (#parties = {n_parties})")
    plt.show()

    plt.figure()
    corr1_order = np.argsort(party_corr1_summary)
    plt.scatter(party_corr1_summary[corr1_order], acc_summary[corr1_order], marker='o')
    # add a regression line
    z = np.polyfit(party_corr1_summary[corr1_order], acc_summary[corr1_order], 2)
    py = np.polyval(z, party_corr1_summary[corr1_order])
    plt.plot(party_corr1_summary[corr1_order], py, "r--")
    plt.xlabel("Party correlation")
    plt.ylabel("Accuracy")
    plt.title(f"Relative party correlation vs accuracy (#parties = {n_parties})")
    plt.show()

    # # summarize the importance, correlation and accuracy into a latex table
    # print("Party & Importance & Correlation & Accuracy \\\\")
    # for i in range(1, len(Xs_train)):
    #     print(f"{i} & {party_importance[i]:.4f} & {party_corr[i-1]:.4f} & {acc_scores[i-1]:.4f} \\\\")

    print(f"Spearman correlation between party correlation and accuracy: {corr0_acc_spearman:.4f},\n"
          f"Spearman correlation between relative party correlation and accuracy: {corr1_acc_spearman:.4f},\n"
          # f"Spearman correlation between party importance and accuracy: {imp_acc_spearman:.4f}"
          )


def test_corr_and_imp_vs_acc(eval_imp=True, eval_corr=True, n_informative=100, n_redundant=0, n_repeated=195):
    n_parties = 10
    party_corr_summary = []
    party_importance_summary = []
    acc_summary = []
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=1, n_estimators=50, max_depth=5, learning_rate=0.1)
    for seed in range(50):
        X, y = make_classification(n_samples=10000, n_features=200,
                                   n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated,
                                   n_classes=2, random_state=seed, shuffle=True, class_sep=0.5)
        X = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Random split into multiple parties
        primary_splitter = ImportanceSplitter(num_parties=n_parties, weights=[10] + [50] * (n_parties - 1), seed=seed)
        Xs_train_w_primary, Xs_test_w_primary = primary_splitter.split(X_train, X_test)
        X_primary_train = Xs_train_w_primary[0]
        X_primary_test = Xs_test_w_primary[0]
        X_train_secondary = np.concatenate(Xs_train_w_primary[1:], axis=1)
        X_test_secondary = np.concatenate(Xs_test_w_primary[1:], axis=1)
        secondary_splitter = ImportanceSplitter(num_parties=n_parties - 1, weights=[10] * (n_parties - 1), seed=seed)
        Xs_train_secondary, Xs_test_secondary = secondary_splitter.split(X_train_secondary, X_test_secondary)

        # continue only when all the parties has non-zero dimensional X
        if np.any([Xs_train_secondary[i].shape[1] == 0 for i in range(n_parties - 1)]) or X_primary_train.shape[1] == 0:
            print(f"Skipping seed {seed} because some party has zero dimensional X")
            continue

        corr_evaluator = CorrelationEvaluator(gamma=0)

        corr_per_party = []
        imp_per_party = []
        acc_per_party = []
        for i in range(len(Xs_train_secondary)):
            X_train_combine = np.concatenate([X_primary_train, Xs_train_secondary[i]], axis=1)
            X_test_combine = np.concatenate([X_primary_test, Xs_test_secondary[i]], axis=1)
            model.fit(X_train_combine, y_train)
            y_pred = model.predict(X_test_combine)
            acc = accuracy_score(y_test, y_pred)

            acc_per_party.append(acc)
            if eval_imp:
                imp_evaluator = ImportanceEvaluator(model.predict, sample_rate=0.01, seed=seed)
                imp_per_party.append(imp_evaluator.evaluate([X_primary_train, Xs_train_secondary[i]])[1])
            if eval_corr:
                corr_per_party.append(corr_evaluator.evaluate([X_primary_train, Xs_train_secondary[i]]))

        acc_summary.append(acc_per_party)
        if eval_imp:
            print(f"Party importance: {imp_per_party}")
            party_importance_summary.append(imp_per_party)
        if eval_corr:
            print(f"Party correlation: {corr_per_party}")
            party_corr_summary.append(corr_per_party)

    party_corr_summary = np.array(party_corr_summary).flatten()
    party_importance_summary = np.array(party_importance_summary).flatten()
    acc_summary = np.array(acc_summary).flatten()

    if eval_imp:
        imp_acc_spearman = spearmanr(party_importance_summary, acc_summary).correlation
        print(f"Spearman correlation between party importance and accuracy: {imp_acc_spearman:.4f}")
    if eval_corr:
        corr_acc_spearman = spearmanr(party_corr_summary, acc_summary).correlation
        print(f"Spearman correlation between party correlation and accuracy: {corr_acc_spearman:.4f}")

    # plot the correlation and importance vs accuracy
    if eval_imp:
        imp_order = np.argsort(party_importance_summary)
        plt.figure()
        plt.scatter(party_importance_summary, acc_summary, marker='o')
        # add a regression line
        z = np.polyfit(party_importance_summary, acc_summary, 1)
        py = np.polyval(z, party_importance_summary[imp_order])
        plt.plot(party_importance_summary[imp_order], py, "r--")
        plt.xlabel("Party importance")
        plt.ylabel("Accuracy")
        plt.title("Party importance vs accuracy")
        plt.show()

    if eval_corr:
        corr_order = np.argsort(party_corr_summary)
        plt.figure()
        plt.scatter(party_corr_summary, acc_summary, marker='o')
        # add a regression line
        z = np.polyfit(party_corr_summary, acc_summary, 1)
        py = np.polyval(z, party_corr_summary[corr_order])
        plt.plot(party_corr_summary[corr_order], py, "r--")
        plt.xlabel("Party correlation")
        plt.ylabel("Accuracy")
        plt.title("Party correlation vs accuracy")
        plt.show()


def test_corr_vs_delta_acc(n_rounds=1, n_informative=5, n_redundant=0, n_repeated=0, seed=0):
    X, y = make_classification(n_samples=10000, n_features=50,
                               n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated,
                               n_classes=2, random_state=seed, shuffle=True, class_sep=1)
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=1, n_estimators=50, max_depth=5, learning_rate=0.3)
    train_ids, test_ids = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=seed)

    # train the model on the whole data
    model.fit(X[train_ids], y[train_ids])
    y_pred = model.predict(X[test_ids])
    acc_combine = accuracy_score(y[test_ids], y_pred)

    np.random.seed(seed)
    corr_summary = []
    acc1_summary = []
    acc2_summary = []
    corr_evaluator = CorrelationEvaluator(gamma=0)
    corr_splitter = CorrelationSplitter(num_parties=2, evaluator=corr_evaluator, seed=seed)
    corr_splitter.fit(X, verbose=True)
    for beta in np.arange(0, 1, 0.1):
        for i in range(n_rounds):
            print(f"Round {i}, beta={beta}")
            # split the columns into two parties
            X1, X2 = corr_splitter.split(X, beta=beta, verbose=False)

            # predefined train/test split
            X1_train, X1_test, X2_train, X2_test = X1[train_ids], X1[test_ids], X2[train_ids], X2[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]

            # train the model on X1 and X2, respectively
            model.fit(X1_train, y_train)
            y_pred1 = model.predict(X1_test)
            acc1 = accuracy_score(y_test, y_pred1)
            model.fit(X2_train, y_train)
            y_pred2 = model.predict(X2_test)
            acc2 = accuracy_score(y_test, y_pred2)

            # calculate the inter-party correlation
            corr = corr_evaluator.evaluate([X1_train, X2_train])

            corr_summary.append(corr)
            acc1_summary.append(acc1)
            acc2_summary.append(acc2)
            print(f"Correlation: {corr:.4f}, accuracy on party 1: {acc1:.4f}, accuracy on party 2: {acc2:.4f}")

    # plot the correlation vs accuracy improvement
    corr_summary = np.array(corr_summary)
    acc1_summary = np.array(acc1_summary)
    acc2_summary = np.array(acc2_summary)
    mean_acc_improve = acc_combine - (acc1_summary + acc2_summary) / 2

    corr_order = np.argsort(corr_summary)
    plt.figure()
    plt.scatter(corr_summary[corr_order], mean_acc_improve[corr_order], marker='o')
    # add a regression line
    z = np.polyfit(corr_summary[corr_order], mean_acc_improve[corr_order], 1)
    py = np.polyval(z, corr_summary[corr_order])
    plt.plot(corr_summary[corr_order], py, "r--")
    plt.xlabel("Inter-party correlation")
    plt.ylabel("Average accuracy improvement")
    plt.title("Inter-party correlation vs average accuracy improvement")
    plt.show()

    # plt.figure()
    # plt.scatter(corr_summary, acc_combine - acc2_summary, marker='o')
    # # add a regression line
    # z = np.polyfit(corr_summary, acc_combine - acc2_summary, 1)
    # py = np.polyval(z, corr_summary[corr_order])
    # plt.plot(corr_summary[corr_order], py, "r--")
    # plt.xlabel("Inter-party correlation")
    # plt.ylabel("Accuracy improvement")
    # plt.title("Inter-party correlation vs accuracy improvement on party 2")
    # plt.show()


def test_imp_vs_delta_acc(n_rounds=1, n_informative=5, n_redundant=0, n_repeated=0, seed=0):
    X, y = make_classification(n_samples=10000, n_features=50,
                               n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated,
                               n_classes=2, random_state=seed, shuffle=True, class_sep=1)
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=1, n_estimators=50, max_depth=5, learning_rate=0.3)
    train_ids, test_ids = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=seed)

    # train the model on the whole data
    model.fit(X[train_ids], y[train_ids])
    y_pred = model.predict(X[test_ids])
    acc_combine = accuracy_score(y[test_ids], y_pred)

    # evaluate the importance of each feature
    imp_evaluator = ImportanceEvaluator(sample_rate=0.005, seed=seed)
    imp_by_feature = imp_evaluator.evaluate_feature(X, model.predict)

    np.random.seed(seed)
    imp1_summary = []
    imp2_summary = []
    acc1_summary = []
    acc2_summary = []
    imp_splitter = ImportanceSplitter(num_parties=2, seed=seed, weights=1)

    for i in range(n_rounds):
        # split the columns into two parties
        (id1, id2) = imp_splitter.split_indices(X)
        imp1 = imp_by_feature[id1].sum()
        imp2 = imp_by_feature[id2].sum()
        imp1_summary.append(imp1)
        imp2_summary.append(imp2)

        # predefined train/test split
        X1, X2 = imp_splitter.split(X, indices=(id1, id2))
        X1_train, X1_test, X2_train, X2_test = X1[train_ids], X1[test_ids], X2[train_ids], X2[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        # train the model on X1 and X2, respectively
        model.fit(X1_train, y_train)
        y_pred1 = model.predict(X1_test)
        acc1 = accuracy_score(y_test, y_pred1)
        model.fit(X2_train, y_train)
        y_pred2 = model.predict(X2_test)
        acc2 = accuracy_score(y_test, y_pred2)
        acc1_summary.append(acc1)
        acc2_summary.append(acc2)
        print(f"Round {i}, importance on party 1: {imp1:.4f}, importance on party 2: {imp2:.4f}, "
                f"accuracy on party 1: {acc1:.4f}, accuracy on party 2: {acc2:.4f}")

    # plot the importance vs accuracy improvement
    imp_summary = np.array(imp1_summary + imp2_summary)
    acc_summary = np.array(acc1_summary + acc2_summary)
    mean_acc_improve = acc_combine - acc_summary / 2

    imp_order = np.argsort(imp_summary)
    plt.figure()
    plt.scatter(imp_summary[imp_order], mean_acc_improve[imp_order], marker='o')
    # add a regression line
    z = np.polyfit(imp_summary[imp_order], mean_acc_improve[imp_order], 1)
    py = np.polyval(z, imp_summary[imp_order])
    plt.plot(imp_summary[imp_order], py, "r--")
    plt.xlabel("Feature importance")
    plt.ylabel("Average accuracy improvement")
    plt.title("Feature importance vs average accuracy improvement")
    plt.show()








if __name__ == '__main__':
    # test_importance_splitter_same_alpha(2000)
    # test_importance_splitter_diff_alpha(2000)
    # test_weight_different_alpha()
    # test_corr_splitter()

    # test_corr_vs_delta_acc(n_rounds=1, n_informative=50, n_redundant=0, n_repeated=0, seed=0)
    # test_corr_vs_delta_acc(n_rounds=1, n_informative=5, n_redundant=45, n_repeated=0, seed=0)
    # test_corr_vs_delta_acc(n_rounds=1, n_informative=5, n_redundant=0, n_repeated=45, seed=0)
    # test_corr_vs_delta_acc(n_rounds=1, n_informative=5, n_redundant=0, n_repeated=0, seed=0)

    test_imp_vs_delta_acc(n_rounds=5, n_informative=50, n_redundant=0, n_repeated=0, seed=0)
    test_imp_vs_delta_acc(n_rounds=5, n_informative=5, n_redundant=45, n_repeated=0, seed=0)
    test_imp_vs_delta_acc(n_rounds=5, n_informative=5, n_redundant=0, n_repeated=45, seed=0)
    test_imp_vs_delta_acc(n_rounds=5, n_informative=5, n_redundant=0, n_repeated=0, seed=0)
