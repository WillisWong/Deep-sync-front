"""
Read raw data
Load/Save binary dataset
"""

import os
try:
    # python >= 3.9
    import pickle
except ImportError:
    # python < 3.9
    import pickle5 as pickle

from alive_progress import alive_bar

from dataset.preprocess.Object import ObjectCategories
from Setting import *


def read_raw_data(data_folder):
    data_loaded = {}
    data_dir = f"{processedData}/{data_folder}"
    data_loaded['data_dir'] = data_dir
    data_loaded['category_map'] = ObjectCategories()

    files = os.listdir(data_dir)
    files = [f for f in files if ".pkl" in f and "domain" not in f]
    cats = []

    with open(f"{data_dir}/final_categories_frequency", "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            a = temp[0]
            for q in temp[1:-1]:
                a = a + ' ' + q
            cats.append(a)

    data_loaded['categories'] = [cat for cat in cats if cat not in {'window', 'door', '5'}]
    data_loaded['cat_to_index'] = {data_loaded['categories'][i] : i for i in range(len(data_loaded['categories']))}

    with open(f"{data_dir}/model_frequency", "r") as f:
        lines = f.readlines()
        models = [line.split()[0] for line in lines]
        data_loaded['model_freq'] = [int(l[:-1].split()[1]) for l in lines]
    
    data_loaded['models'] = [
        model for model in models if data_loaded['category_map'].get_final_category(model) not in {'window', 'door', ''}
        ]
    data_loaded['model_to_index'] = {
        data_loaded['models'][i]: i for i in range(len(data_loaded['models']))
        }

    N = len(data_loaded['models'])
    data_loaded['num_categories'] = len(data_loaded['categories'])

    data_loaded['model_index_to_cat'] = [
        data_loaded['cat_to_index'][data_loaded['category_map'].get_final_category(data_loaded['models'][i])] for i in range(N)
    ]

    data_loaded['count'] = [
        [0 for i in range(N)] for j in range(N)
    ]
    with alive_bar(len(files)) as bar:
        for index in range(len(files)):
            with open(f"{data_dir}/{index}.pkl", "rb") as f:
                # print(pickle.load(f))
                (_, _, nodes), _ = pickle.load(f)

            object_nodes = []
            for node in nodes:
                modelId = node["modelId"]
                category = data_loaded['category_map'].get_final_category(modelId)
                # print(category)
                if category not in ["door", "window", '']:
                    object_nodes.append(node)
            print(data_loaded['model_to_index'])
            for i in range(len(object_nodes)):
                for j in range(i+1, len(object_nodes)):
                    a = data_loaded['model_to_index'][object_nodes[i]["modelId"]]
                    b = data_loaded['model_to_index'][object_nodes[j]["modelId"]]
                    data_loaded['count'][a][b] += 1
                    data_loaded['count'][b][a] += 1
            # print(index, end="\r")
            bar()

    data_loaded['N'] = N

    return data_loaded, data_dir


def save_binary_dataset(data, destination=None):
    if destination is None:
        print("No Dir found !")
        return 0
    filename = f"{destination}/model_prior.pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_binary_dataset(data_dir):
    source = f"{data_dir}/model_prior.pkl"
    with open(source, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    data, dest = read_raw_data('bedroom_new')
    save_binary_dataset(data, dest)