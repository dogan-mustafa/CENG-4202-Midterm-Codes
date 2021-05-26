import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

parameters = [["gini", 10], ["gini", 50], ["gini", 100], ["gini", 200],
              ["entropy", 10], ["entropy", 50], ["entropy", 100], ["entropy", 200]]
best_parameters = []
best_scores = [0, 0]
scores = []

for param in parameters:

    clf = DecisionTreeClassifier(
        criterion=param[0], splitter='random', max_depth=param[1], class_weight='balanced')  # Train Model.
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)

    acc_score = accuracy_score(y_val, y_val_pred)
    f1_scr = f1_score(y_val, y_val_pred, average='weighted')
    scores.append([acc_score, f1_score])

    if(acc_score > best_scores[0]):
        best_scores = [acc_score, f1_scr]
        best_parameters = param

print("Best Test Accuracy Score: ", best_scores[0])
print("Best Test F1 Score: ", best_scores[1])
print("Best Parameters (Maximum Depth, Learning Rate): ", best_parameters)


clf = DecisionTreeClassifier(
    criterion=best_parameters[0], splitter='random', max_depth=best_parameters[1], class_weight='balanced')  # Final Test.
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_test_pred)
f1_scr = f1_score(y_test, y_test_pred, average='weighted')

print("Final Accracy Score:", acc_score)
print("Final F1 Score", f1_scr)
