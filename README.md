# MLFramework

## Python machine learning framework for classification problemns.

This is a developing version and the following algorithms have already implemented:

- Scikit-learn:
    - Logistic Regression (lr)
    - Decision Tree (dt)
    - Random Forest (rf)
    - Support Vector Machine (svm)
    - Artificial Neural Network (ann)
    - K-Nearest Neighbors (knn)

- XGboost (xgb)

### How to start

Best practices is to create a Python environment and install the required packages:

```python
python3 -m pip install --user virtualenv # if you do not have the virtualenv
git clone https://github.com/dtiezzi/MLFramework.git
cd MLFramework
which python
virtualenv --python"/path/to/your/python/path" env
source env/bin/activate
python -m pip install -r requirements.txt
```

You have to move your CSV file into the `static` folder. Then, in order to run the ML pipeline:

```python
python main.py
```

You can use the Scikit-learn breast cancer sample dataset as an example. The pipeline outputs accuracy graphs and records:

```
Decision Tree:
	Brier: 0.096
	Precision: 0.901
	Recall: 0.951
	F1: 0.925
```

When running the Decision Tree algorithm, the the decision tree plot is created:

<img src="_img/test_decision_tree.svg" style="display: block; margin-left: auto; margin-right: auto;">

All algorithms save a ROC curve, a Reliability plot and the Confusion Matrix:

<img src="_img/test_DT_ROC.png" height='350px' style="display: block; margin-left: auto; margin-right: auto; width: 50%;">
</br>
Calibration Plot:
<img src="_img/test_DT_CP.png" height='350px' style="display: block; margin-left: auto; margin-right: auto; width: 50%;">
</br>
Confusion Matrix:
<img src="_img/test_DT_CM.png" height='325px' style="display: block; margin-left: auto; margin-right: auto; width: 50%;">

The SMOTE oversampling method is available and is recommended to use for unbalanced samples. You can use the `IterativeImputer` class from the Scikit-learn package to impute missing values.

You can use the Grid Search option to automatically tune the hyperparameters. Predefined parameters for searching are in the `gridsc.py` and `rfparams.py` files. You may change the parameters if needed.

Plots are saved into `reports/graphs` folders. Your models and the accuracy report are saved in `reports/models` and `reports/txtfiles`, respectively.
