import pickle
import os

def write_pickle(item, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(item, file)

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)