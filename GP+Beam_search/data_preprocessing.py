import re
import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

Inverse_trig = {
    'arcsin': 'asin',
    'arccos': 'acos',
    'arctan': 'atan',
    'arccot': 'acot',
    'arcsec': 'asec',
    'arccsc': 'acsc',
    'arcsinh': 'asinh',
    'arccosh': 'acosh',
    'arctanh': 'atanh',
    'arccoth': 'acoth',
    'arcsech': 'asech',
    'arccsch': 'acsch',         
}

def load_and_clean_data(file_path):
    df_target = pd.read_csv(file_path)
    df_target = df_target.dropna(subset=['Filename'])

    # Correct specific entries in the dataset
    corrections = {
        21: 3, 22: 4, 38: 4, 82: 3, 90: 4, 94: 3, 99: 3,
        102: 4, 116: 4, 129: 4, 130: 3, 137: 4, 154: 3, 
        160: 4, 165: 3, 166: 4
    }
    
    for index, value in corrections.items():
        df_target.loc[index, '# variables'] = value

    for i in range(len(df_target)):
        formula = df_target['Formula'][i]
        for a in Inverse_trig.keys():
            df_target.loc[i,'Formula'] = re.sub(a,Inverse_trig[a],formula)

    return df_target

class FeynmanDataset(Dataset):
    def __init__(self, df, dataset_dir):
        super().__init__()
        self.df = df
        self.dataset_dir = dataset_dir
        self.prefix_equations = np.load(os.path.join(dataset_dir, "prefix_equations.npy"))
        # prefix_equations = []

        prefix_equations = []
        for prefix in self.prefix_equations:
            prefix_equations.append(np.trim_zeros(prefix))

        self.prefix_equations = prefix_equations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(os.path.join(self.dataset_dir, row['Filename']), f"{row['data_num']}.npy")
        x = np.load(path).astype(np.int32)

        path = os.path.join(self.dataset_dir, f"{row['Filename']}.npy")
        y = self.prefix_equations[int(row['number']) - 1]

        return (torch.Tensor(x).long(), torch.Tensor(y).long())
    
def get_datasets(df, input_df, dataset_dir):
    train_df, test_df = train_test_split(df, test_size=0.1,random_state = 42)
    train_equations = train_df['Filename'].tolist()
    test_equations = test_df['Filename'].tolist()

    input_test_df = input_df[input_df['Filename'].isin(test_equations)]
    input_train_df = input_df[input_df['Filename'].isin(train_equations)]

    input_train_df, input_val_df = train_test_split(input_train_df, test_size = 0.1, shuffle=True)

    train_dataset = FeynmanDataset(input_train_df, dataset_dir)
    val_dataset = FeynmanDataset(input_val_df, dataset_dir)
    test_dataset = FeynmanDataset(input_test_df, dataset_dir)

    datasets = {
        "train":train_dataset,
        "test":test_dataset,
        "valid":val_dataset
        }

    return datasets

def load_data(config, file_index):
    with open(config.input_path + '/' + config.test_file_paths[file_index]) as file:
        data = file.readlines()

    arr = np.array([i.split() for i in data], dtype=np.float32)

    points = []
    num_vars = 0
    for i in arr:
        count = 0
        temp = []
        for j in i[:-1]:
            count += 1
            temp.append(j)
        num_vars = count
        points.append((temp, i[-1]))
    
    points = points[:2000]
    return points, num_vars

def preprocess_data(config):
    train_df = pd.read_csv(config.train_df_path)
    train_df.rename(columns={'filename': 'Filename'}, inplace=True)
    df_target = pd.read_csv(config.df_target_path)
    # df_target = load_and_clean_data(config.df_target_path)
    datasets = get_datasets(df_target, train_df, config.dataset_arrays_path)
    return train_df, df_target, datasets
