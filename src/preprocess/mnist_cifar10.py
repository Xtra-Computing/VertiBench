import numpy as np
import torchvision
import pandas as pd

def get_mnist_data():
    train = torchvision.datasets.MNIST('./data/syn/mnist', train=True, download=True)
    test = torchvision.datasets.MNIST('./data/syn/mnist', train=False, download=True)
    return train, test

def get_cifar10_data():
    train = torchvision.datasets.CIFAR10('./data/syn/cifar10', train=True, download=True)
    test = torchvision.datasets.CIFAR10('./data/syn/cifar10', train=False, download=True)
    return train, test

def get_data_as_df(train, test):
    train_data = train.data.numpy()
    train_labels = train.targets.numpy()
    test_data = test.data.numpy()
    test_labels = test.targets.numpy()
    train_data = train_data.reshape(-1, 28*28)
    test_data = test_data.reshape(-1, 28*28)
    df_train = pd.DataFrame(train_data)
    df_train.columns = [f'x{i}' for i in range(28*28)]
    df_train['id'] = df_train.index
    df_train['y'] = train_labels
    cols = df_train.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_train = df_train[cols]

    df_test = pd.DataFrame(test_data)
    df_test.columns = [f'x{i}' for i in range(28*28)]
    df_test['id'] = df_test.index
    df_test['y'] = test_labels
    cols = df_test.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_test = df_test[cols]

    return df_train, df_test

def save_data(df_train, df_test):
    df_train.to_csv('./data/syn/mnist/mnist_train.csv', index=False, header=True, sep=',')
    df_test.to_csv('./data/syn/mnist/mnist_test.csv', index=False, header=True, sep=',')

train, test = get_mnist_data()
df_train, df_test = get_data_as_df(train, test)
save_data(df_train, df_test)

train, test = get_cifar10_data()



train_data = np.array(train.data)
train_label = np.array(train.targets)
test_data = np.array(test.data)
test_label = np.array(test.targets)

df_train = pd.DataFrame(train_data.reshape(-1, 32*32*3))
df_train.columns = [f'x{i}' for i in range(32*32*3)]
df_train['id'] = df_train.index
df_train['y'] = train_label
cols = df_train.columns.tolist()
cols = cols[-2:] + cols[:-2]
df_train = df_train[cols]

df_test = pd.DataFrame(test_data.reshape(-1, 32*32*3))
df_test.columns = [f'x{i}' for i in range(32*32*3)]
df_test['id'] = df_test.index
df_test['y'] = test_label
cols = df_test.columns.tolist()
cols = cols[-2:] + cols[:-2]
df_test = df_test[cols]

df_train.to_csv('./data/syn/cifar10/cifar10_train.csv', index=False, header=True, sep=',')
df_test.to_csv('./data/syn/cifar10/cifar10_test.csv', index=False, header=True, sep=',')