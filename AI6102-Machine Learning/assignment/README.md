# AI6102 Assignment

Individual assignment for AI6102 Machine Learning Methodologies & Applications (NTU, 2026).

| File | Content |
|---|---|
| `assignment.pdf` | Assignment questions |
| `assignment_solution.pdf` / `.tex` | Full written solution (LaTeX source included) |
| `solve_q2.py` | Code for Question 2 (SVM experiments on a9a) |
| `q2_results.json` | Cross-validation results produced by `solve_q2.py` |
| `a9a`, `a9a.t` | LIBSVM a9a dataset (train / test), as used in Question 2 |
| `AI6102_Final_Exam_Preparation/` | Final-exam revision package built during the course (see its README) |

## Solution overview

1. **Multi-class regularised logistic regression**: regularised negative log-likelihood, gradient with respect to each class weight vector, and gradient-descent update rules.
2. **SVM experiments on a9a with scikit-learn**: 3-fold cross-validation over C for a linear kernel and over a 5 x 5 (C, gamma) grid for an RBF kernel, followed by test-set evaluation of the best settings.
3. **Soft-margin SVM as structural risk minimisation**: algebraic equivalence between the primal soft-margin SVM and hinge loss with L2 regularisation.
4. **Kernelised ridge regression**: representer theorem and the kernel form of the solution.

## Running Question 2

```bash
pip install numpy scikit-learn
python solve_q2.py
```

The script uses the local `a9a` / `a9a.t` files; if they are missing it downloads them from the LIBSVM dataset page. The full grid search takes a while because `SVC` is trained many times on about 32,000 samples.
