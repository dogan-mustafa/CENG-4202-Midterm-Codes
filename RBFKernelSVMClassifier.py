import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Loading Data
data = pd.read_csv("TrainingData.csv")

# Preprocessing Data
data = data.drop(columns=["Id"], axis=1)  # Drop IDs.

ordinalEncoder = OrdinalEncoder()  # Decode Categorical Data ordinaly.
data = pd.DataFrame(ordinalEncoder.fit_transform(data), columns=list(data))

# Drop Labels and store in a different variable.
x = data.drop(columns=['risk_flag'], axis=1)
y = data['risk_flag']

smooter = SMOTE()  # Data is skewed. So, smoote it.
x, y = smooter.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

# Create Parameter set and start to train.
C_param_list = np.logspace(-2, 10, 13)
Gamma_param_list = np.logspace(-9, 3, 13)
params = dict(gamma=Gamma_param_list, C=C_param_list)
cross_val = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
rbf_classfier = GridSearchCV(SVC(), param_grid=params, cv=cross_val)
rbf_classfier.fit(X_train, y_train)

print("Best Parameters (C and Gamma): ", rbf_classfier.best_params_)
print("Best Test Accuracy Score: ", rbf_classfier.best_score_)

# The process of tuning parameters takes long time. So I did not use test and validation sets.
