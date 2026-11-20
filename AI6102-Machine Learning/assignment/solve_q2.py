"""
AI6102 Assignment - Question 2: SVM Experiments on a9a dataset
"""

import numpy as np
import urllib.request
import ssl
import os

# Bypass SSL verification (LIBSVM server has an incomplete cert chain)
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. Download a9a dataset from LIBSVM
# ─────────────────────────────────────────────
BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
FILES = {"a9a": "a9a", "a9a.t": "a9a.t"}

for fname, url_name in FILES.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname} ...")
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        with opener.open(BASE_URL + url_name) as resp, open(fname, "wb") as out:
            out.write(resp.read())
        print(f"  Saved {fname}")
    else:
        print(f"  {fname} already exists, skipping download.")

# ─────────────────────────────────────────────
# 2. Load data (force n_features=123)
# ─────────────────────────────────────────────
print("\nLoading datasets...")
X_train, y_train = load_svmlight_file("a9a",   n_features=123)
X_test,  y_test  = load_svmlight_file("a9a.t", n_features=123)
X_train = X_train.toarray()
X_test  = X_test.toarray()
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 3. Cross-validation setup
# ─────────────────────────────────────────────
kf = KFold(n_splits=3, shuffle=True, random_state=42)
C_values     = [0.01, 0.05, 0.1, 0.5, 1]
gamma_values = [0.01, 0.05, 0.1, 0.5, 1]

# ─────────────────────────────────────────────
# Table 1: Linear Kernel
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE 1: Linear Kernel — Mean Validation Accuracy (3-fold CV)")
print("="*60)
print(f"{'C':>6}  {'Mean Acc':>10}")
print("-"*20)

linear_results = {}
for C in C_values:
    clf = SVC(kernel="linear", C=C)
    scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)
    mean_acc = scores.mean()
    linear_results[C] = mean_acc
    print(f"{C:>6.2f}  {mean_acc:>10.6f}")

best_linear_C   = max(linear_results, key=linear_results.get)
best_linear_acc = linear_results[best_linear_C]
print(f"\nBest Linear: C={best_linear_C}, Mean Val Acc={best_linear_acc:.6f}")

# ─────────────────────────────────────────────
# Table 2: RBF Kernel — 5×5 Grid Search
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE 2: RBF Kernel — Mean Validation Accuracy (3-fold CV)")
print("="*60)

# header
header = f"{'C\\gamma':>8}" + "".join(f"{g:>10.2f}" for g in gamma_values)
print(header)
print("-" * (8 + 10 * len(gamma_values)))

rbf_results = {}
for C in C_values:
    row_str = f"{C:>8.2f}"
    for gamma in gamma_values:
        clf = SVC(kernel="rbf", C=C, gamma=gamma)
        scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)
        mean_acc = scores.mean()
        rbf_results[(C, gamma)] = mean_acc
        row_str += f"{mean_acc:>10.6f}"
    print(row_str)

best_rbf_key = max(rbf_results, key=rbf_results.get)
best_rbf_C, best_rbf_gamma = best_rbf_key
best_rbf_acc = rbf_results[best_rbf_key]
print(f"\nBest RBF: C={best_rbf_C}, gamma={best_rbf_gamma}, Mean Val Acc={best_rbf_acc:.6f}")

# ─────────────────────────────────────────────
# Table 3: Test Set Evaluation
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE 3: Test Set Evaluation")
print("="*60)

# Determine global best
if best_rbf_acc >= best_linear_acc:
    best_kernel = "rbf"
    best_params = {"C": best_rbf_C, "gamma": best_rbf_gamma}
    best_val_acc = best_rbf_acc
    print(f"Best overall: RBF kernel, C={best_rbf_C}, gamma={best_rbf_gamma}")
else:
    best_kernel = "linear"
    best_params = {"C": best_linear_C}
    best_val_acc = best_linear_acc
    print(f"Best overall: Linear kernel, C={best_linear_C}")

# Retrain on entire training set
print("Retraining on full training set...")
if best_kernel == "rbf":
    final_clf = SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"])
else:
    final_clf = SVC(kernel="linear", C=best_params["C"])

final_clf.fit(X_train, y_train)
y_pred = final_clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n{'Kernel':>10}  {'Best C':>8}  {'Best gamma':>12}  {'Val Acc':>10}  {'Test Acc':>10}")
print("-" * 60)
if best_kernel == "rbf":
    print(f"{'RBF':>10}  {best_rbf_C:>8.2f}  {best_rbf_gamma:>12.2f}  {best_val_acc:>10.6f}  {test_acc:>10.6f}")
else:
    print(f"{'Linear':>10}  {best_linear_C:>8.2f}  {'N/A':>12}  {best_val_acc:>10.6f}  {test_acc:>10.6f}")

# ─────────────────────────────────────────────
# Save results for LaTeX
# ─────────────────────────────────────────────
import json
results_export = {
    "linear": {str(C): linear_results[C] for C in C_values},
    "rbf": {f"{C},{g}": rbf_results[(C, g)] for C in C_values for g in gamma_values},
    "best_kernel": best_kernel,
    "best_C": best_params["C"],
    "best_gamma": best_params.get("gamma", None),
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
    "C_values": C_values,
    "gamma_values": gamma_values,
}
with open("q2_results.json", "w") as f:
    json.dump(results_export, f, indent=2)
print("\nResults saved to q2_results.json")
