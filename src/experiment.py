from main import tree_grow, tree_pred, tree_grow_b, tree_pred_b
from load_data import get_eclipse_2, get_eclipse_3
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Tuple, List, Dict
from scipy.stats import chi2


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    y_correctness = y_true == y_pred

    return accuracy, precision, recall, f1, cm, y_correctness


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    x_train, y_train, feature_names = get_eclipse_2()
    x_test, y_test, _ = get_eclipse_3()
    return x_train, y_train, x_test, y_test, feature_names


def experiment1() -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    x_train, y_train, x_test, y_test, feature_names = load_data()

    tree = tree_grow(x_train, y_train, nmin=15, minleaf=5, nfeat=41)
    y_pred = tree_pred(x_test, tree)

    print("Running single decision tree with nmin=15, minleaf=5...")

    return calculate_metrics(y_test, y_pred)


def experiment2() -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    x_train, y_train, x_test, y_test, _ = load_data()

    trs = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
    y_pred = tree_pred_b(x_test, trs)

    print("Running 100 trees with bagging using all features with nmin=15, minleaf=5...")

    return calculate_metrics(y_test, y_pred)


def experiment3() -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    x_train, y_train, x_test, y_test, _ = load_data()

    trs = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=6, m=100)
    y_pred = tree_pred_b(x_test, trs)

    print("Running random forests using 100 trees with nmin=15, minleaf=5...")

    return calculate_metrics(y_test, y_pred)


def run_experiments() -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]: 
    a1, p1, r1, f1, cm1, ycorrectness1 = experiment1()
    a2, p2, r2, f2, cm2, ycorrectness2 = experiment2()
    a3, p3, r3, f3, cm3, ycorrectness3 = experiment3()

    metrics = np.array([[a1, p1, r1, f1], [a2, p2, r2, f2], [a3, p3, r3, f3]])
    return metrics, [cm1, cm2, cm3], [ycorrectness1, ycorrectness2, ycorrectness3]


def calculate_significance(ycorrectness: List[np.ndarray]) -> np.ndarray:
    ct_tree_bagging = confusion_matrix(y_true=ycorrectness[0], y_pred=ycorrectness[1])
    ct_tree_rf = confusion_matrix(y_true=ycorrectness[0], y_pred=ycorrectness[2])
    ct_bagging_rf = confusion_matrix(y_true=ycorrectness[1], y_pred=ycorrectness[2])
     
    res_t_b = chi2_test(ct_tree_bagging)
    res_t_rf = chi2_test(ct_tree_rf)
    res_b_rf = chi2_test(ct_bagging_rf)

    return np.array([[res_t_b[0], res_t_b[1]], [res_t_rf[0], res_t_rf[1]], [res_b_rf[0], res_b_rf[1]]])


def chi2_test(contingency_matrix: np.ndarray) -> Tuple[float, float]:
    b = contingency_matrix[0,1]
    c = contingency_matrix[1,0]
    chi2_value = (b-c)**2 / (b+c)
    return chi2_value, chi2.sf(chi2_value, df=1)


def report_experiments(scores: np.ndarray, cms: List[np.ndarray], ycorrectness: List[np.ndarray]):
    for i in range(1, 4):
        print(f"The results of experiment {i}:")
        print(f"Accuracy: {scores[i-1, 0]}, Precision: {scores[i-1, 1]}, Recall: {scores[i-1, 2]}, F1: {scores[i-1, 3]}")
        print("Confusion matrix:")
        print(cms[i-1])
        print()

    significance_scores = calculate_significance(ycorrectness)
    print("Results of McNemar's test between single tree and bagging with m=100")
    print(f"chi2-stat: {significance_scores[0, 0]}, p-val: {significance_scores[0, 1]}")
    print()
    print("Results of McNemar's test single tree and random forests with m=100")
    print(f"chi2-stat: {significance_scores[1, 0]}, p-val: {significance_scores[1, 1]}")
    print()
    print("Results of McNemar's test bagging with m=100 and random forests with m=100")
    print(f"chi2-stat: {significance_scores[2, 0]}, p-val: {significance_scores[2, 1]}")


scores, cms, ycorrectness = run_experiments()
report_experiments(scores=scores, cms=cms, ycorrectness=ycorrectness)