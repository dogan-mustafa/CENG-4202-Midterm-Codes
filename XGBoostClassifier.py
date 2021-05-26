import pandas as pd
import xgboost
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

parameters = [[5, 0.001], [10, 0.001], [15, 0.001], [5, 0.01],
              [10, 0.01], [15, 0.01], [5, 0.1], [10, 0.1], [15, 0.1]]
best_parameters = []
best_scores = [0, 0]
scores = []

for param in parameters:

    xgb = xgboost.XGBClassifier(
        enable_categorical=False, max_depth=param[0], learning_rate=param[1])  # Train Model.
    xgb.fit(X_train, y_train)

    y_val_pred = xgb.predict(X_val)

    acc_score = accuracy_score(y_val, y_val_pred)
    f1_scr = f1_score(y_val, y_val_pred, average='weighted')
    scores.append([acc_score, f1_score])

    if(acc_score > best_scores[0]):
        best_scores = [acc_score, f1_scr]
        best_parameters = param

print("Best Test Accuracy Score: ", best_scores[0])
print("Best Test F1 Score: ", best_scores[1])
print("Best Parameters (Maximum Depth, Learning Rate): ", best_parameters)


xgb = xgboost.XGBClassifier(
    enable_categorical=False, max_depth=best_parameters[0], learning_rate=best_parameters[1])  # Final Test.
xgb.fit(X_train, y_train)

y_test_pred = xgb.predict(X_test)
acc_score = accuracy_score(y_test, y_test_pred)
f1_scr = f1_score(y_test, y_test_pred, average='weighted')

print("Final Accracy Score:", acc_score)
print("Final F1 Score", f1_scr)

my_confusion_matrix = confusion_matrix(y_test, y_test_pred)

group_labels = ["True Negative", "False Positive",
                "False Negative", "True Positive"]
gorup_label_count = ["{0:0.0f}".format(value) for value in
                     my_confusion_matrix.flatten()]
group_label_percentages = ["{0:.3%}".format(value) for value in
                           my_confusion_matrix.flatten()/np.sum(my_confusion_matrix)]
labels = [f"{a1}\n{a2}\n{a3}" for a1, a2, a3 in
          zip(group_labels, gorup_label_count, group_label_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(my_confusion_matrix, annot=labels, fmt="", cmap='Blues')
