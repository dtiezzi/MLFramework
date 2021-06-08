<img src='_img/LabStamp_logo.png' style="background: white;">

# MLFramework

## Python machine learning framework

This is a developing version and the following algoritms have already implemented:

- Scikit-learn:
    - Logistic Regression (lr)
    - Decision Tree (dt)
    - Random Forest (rf)
    - Support Vector Machine (svm)
    - Artificial Neural Network (ann)

- XGboost (xgb)

### How to start

Best practices is to create a Python environment and install the required packages:

```python
python3 -m pip install --user virtualenv
which python
virtualenv --python"/path/to/your/python/path" env
source env/bin/activate
python -m pip install -r requirements.txt
```

To run the ML pipeline:

```python
python main.py
```


