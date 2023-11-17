import os
import pickle


# 获取zhaomin的数据集
def get_dataset(dataset_name: str, party_id:int, typ:str, value:str, train: str, dataseed: str, num_clients: int):
    print("Get dataset", dataset_name, party_id, typ, value, train, dataseed, num_clients)
    # dataseed: 0,1,2,3,4
    if typ =="corr":
        value = "beta" + value
    elif typ == "imp": # 0.1，1.0，10.0，100.0
        value = "weight" + value
    else:
        assert False
    
    fname = ""
    if dataset_name in ['mnist', 'cifar10']:
        fname = f"{dataset_name}_{train}_party{num_clients}-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"
    else:
        if train == "train":
            fname = f"{dataset_name}_train_party{num_clients}-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"
        else:
            fname = f"{dataset_name}_test_party{num_clients}-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"

    msd = pickle.load(open(f"../../../data/syn/{dataset_name}/{fname}", "rb")) # cd to VertiBench root dir
    return msd

# 获取zhaomin的数据集
def get_dataset_real(dataset_name: str, party_id:int, train: str):
    print("Get real dataset", dataset_name)
    
    if dataset_name == "wide":
        num_clients = 5
    elif dataset_name == "vehicle":
        num_clients = 2
    else:
        assert False, "Unknown dataset name"

    fname = ""
    if train == "train":
        fname = f"{dataset_name}_party{num_clients}-{party_id}_train.pkl"
    else:
        fname = f"{dataset_name}_party{num_clients}-{party_id}_test.pkl"

    msd = pickle.load(open(f"../../../data/real/{dataset_name}/processed/{fname}", "rb"))
    return msd