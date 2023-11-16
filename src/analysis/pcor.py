import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def corr_mtx(X1, X2):
    """
    :param X1: iterable of arrays of length d1, each array has shape (n)
    :param X2: iterable of arrays of length d2, each array has shape (n)
    :return: (d, d)
    """
    d1 = len(X1)
    d2 = len(X2)
    corr_matrix = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            assert len(X1[i]) == len(X2[j])
            corr = spearmanr(X1[i], X2[j]).correlation
            corr_matrix[i, j] = corr
    return corr_matrix


def plot_split(X1, X2):

    assert len(X1) == len(X2) == 2
    corr12 = corr_mtx(X1, X2)
    corr11 = corr_mtx(X1, X1)
    corr22 = corr_mtx(X2, X2)

    pcor12 = np.std(np.linalg.svd(corr12)[1], ddof=1) / np.sqrt(2)
    pcor11 = np.std(np.linalg.svd(corr11)[1], ddof=1) / np.sqrt(2)
    pcor22 = np.std(np.linalg.svd(corr22)[1], ddof=1) / np.sqrt(2)

    pcor_mtx = [[pcor11, pcor12], [pcor12, pcor22]]
    Icor = ((pcor12 - pcor11) + (pcor12 - pcor22)) / 2
    print(Icor)
    print(pcor_mtx)

    # plot four scatter plots in a 2x2 grid. Each scatter plot represents a pair of variables.
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Correlation between $A_{0,1}$ and $B_{0,1}$')
    axs[0, 0].scatter(X1[0], X2[0], s=1)
    axs[0, 0].set_title(f'$A_0$ vs. $B_0$')
    axs[0, 1].scatter(X1[0], X2[1], s=1)
    axs[0, 1].set_title(f'$A_0$ vs. $B_1$')
    axs[1, 0].scatter(X1[1], X2[0], s=1)
    axs[1, 0].set_title(f'$A_1$ vs. $B_0$')
    axs[1, 1].scatter(X1[1], X2[1], s=1)
    axs[1, 1].set_title(f'$A_1$ vs. $B_1$')
    plt.tight_layout()
    plt.show()

    # plot pcor matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pcor_mtx, cmap='viridis')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['$A_0$', '$A_1$'])
    ax.set_yticklabels(['$B_0$', '$B_1$'])
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_title('Pcor matrix')
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f"{pcor_mtx[i][j]:.2f}", ha='center', va='center', color='w')
    fig.tight_layout()
    plt.show()

def plot3d(x, y, z, title=None, save_path=None):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

    # plot 3d scatter plot, show depth by color
    fig = plt.figure(figsize=(8, 8))
    # rotate for a better view
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c=z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # add correlation matrix as text (in latex matrix)
    corr = spearmanr(np.stack([x, y, z]).T).correlation
    pcor = np.std(np.linalg.svd(corr)[1], ddof=1) / np.sqrt(3)
    p, svs, q = np.linalg.svd(corr)

    corr_text = rf"Correlation Matrix " \
                rf"=$\begin{{bmatrix}}{corr[0, 0]:.2f} & {corr[0, 1]:.2f} & {corr[0, 2]:.2f} \\ " \
                rf"{corr[1, 0]:.2f} & {corr[1, 1]:.2f} & {corr[1, 2]:.2f} \\ " \
                rf"{corr[2, 0]:.2f} & {corr[2, 1]:.2f} & {corr[2, 2]:.2f} \end{{bmatrix}}$\\"

    ax.text2D(0.3, 0.95, corr_text, transform=ax.transAxes, fontsize=20)

    # add pcor as text
    pcor_text = rf"Pcor={pcor:.2f}, " \
                rf"Singular Values = $\begin{{bmatrix}}{svs[0]:.2f} & {svs[1]:.2f} & {svs[2]:.2f} \end{{bmatrix}}$"
    ax.text2D(0.25, -0.05, pcor_text, transform=ax.transAxes, fontsize=20)

    # add three singular values in the singular vector direction in the 3d plot
    for i in range(3):
        ax.quiver(0.8, -0.5, -0.5, q[0, i], q[1, i], q[2, i],
                  length=svs[i], color='r', arrow_length_ratio=0.3)



    if title is not None:
        # title at left top corner
        ax.set_title(title, x=0.1, y=1.1, ha='left', va='bottom', fontsize=28)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_self_corr(n_points):
    x1 = np.random.uniform(0, 1, n_points)
    y1 = np.random.uniform(0, 1, n_points)
    z1 = np.random.uniform(0, 1, n_points)

    plot3d(x1, y1, z1, "Independent", save_path="fig/pcor-independent-example.png")

    x2 = np.random.uniform(0, 1, n_points)
    y2 = 2 * x2
    z2 = x2 + 1

    plot3d(x2, y2, z2, "Perfect Correlated", save_path="fig/pcor-linear-example.png")

    x3 = np.random.uniform(0, 1, n_points)
    y3 = np.random.uniform(0, 1, n_points)
    z3 = -x3 ** 2 - y3 ** 2

    plot3d(x3, y3, z3, "Partially Correlated", save_path="fig/pcor-partial-example.png")








if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14})
    np.random.seed(0)
    #
    # n_repeat = 10000
    # x1 = np.random.uniform(0, 1, n_repeat)
    # x2 = np.random.uniform(0, 1, n_repeat)
    # x3 = 2 * x1
    # x4 = 2 * x2
    #
    # plot_split([x1, x2], [x3, x4])
    # plot_split([x1, x3], [x2, x4])

    plot_self_corr(10000)
