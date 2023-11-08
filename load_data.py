import pandas as pd
import numpy as np
from typing import Tuple, Dict
import csv

def get_columns_to_use():
    prefixes = ["FOUT", "MLOC", "NBD", "PAR", "VG", "NOF", "NOM", "NSF", "NSM", "ACD", "NOI", "NOT", "TLOC"]
    suffixes = ["avg", "max", "sum"]

    columns_to_use = ["_".join([prefix, suffix]) for prefix in prefixes for suffix in suffixes]
    columns_to_use.extend(["NOCU", "pre"])

    return columns_to_use


def get_eclipse_2() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    return get_eclipse_dataset(version=2)


def get_eclipse_3() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]: 
    return get_eclipse_dataset(version=3)


def get_eclipse_dataset(version: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]: 
    df = pd.read_csv(f"data/eclipse-metrics-packages-{version}.0.csv", delimiter=";")
    feature_names = {k:v for k, v in enumerate(get_columns_to_use())}
    x = (df[get_columns_to_use()]).to_numpy()
    y = (df["post"] > 0).to_numpy(dtype=np.bool8)
    return x, y, feature_names


def get_pima_dataset() -> Tuple[np.ndarray, np.ndarray]:
    df = np.genfromtxt("data/pima.csv", delimiter=",")
    x = df[:, :-1]
    y = df[:, -1]
    return x, y


def get_credit_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    with open("data/credit.csv", "r") as csvfile:
        names = next(csv.reader(csvfile))
        feature_names = {k:v for k, v in enumerate(names[:-1])}
    df = np.genfromtxt("credit.csv", delimiter=",", skip_header=True)
    x = df[:, :-1]
    y = df[:, -1]
    return x, y, feature_names