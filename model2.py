import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.DataFrame({
    "links": [
        5, 0, 3, 1, None, 4, 0, 2, 6, 1,
        0, 5, 2, None, 4, 1, 3, 0, 5, 2,
        7, 1, 0, 6, 2, 4, 1, None, 3, 0,
        5, 2, 8, 1, 4, 0, 6, 3, None, 2,
        1, 5, 0, 4, 7, 2, 3, 1, 6, 0,
        4, None, 2, 5, 1, 8, 0, 3, 6, 2,
        5, 1, 4, 0, 7, 3, None, 2, 5, 1,
        0, 6, 4, 2, 8, 1, 3, 5, 0, None,
        2, 7, 1, 4, 0, 6, 3, 5, 1, 2,
        4, 0, 8, 3, 1, None, 5, 2, 6, 0
    ],
    "words": [
        20, 150, 30, 100, 50, 25, 200, None, 15, 120,
        180, 20, 80, 60, 35, 140, 45, 190, 25, 90,
        45, 210, 35, 160, 70, 25, 130, 55, 18, 220,
        40, 95, 30, 175, 60, 15, 145, 80, 28, 200,
        50, 110, 22, 185, 65, 35, 155, 45, 230, 75,
        25, 100, 40, 170, 55, 20, 135, 85, 32, 195,
        48, 125, 18, 160, 70, 30, 205, 58, 24, 145,
        90, 35, 180, 62, 15, 155, 42, 215, 80, 50,
        120, 28, 190, 68, 38, 165, 52, 225, 75, 20,
        140, 45, 200, 30, 100, 55, 175, 35, 210, 65
    ],
    "sender": [
        "unknown", "known", "unknown", "known", None,
        "unknown", "known", "known", "unknown", "known",
        "known", "unknown", "known", None, "unknown",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "known", "unknown", "known",
        "unknown", "known", None, "unknown", "known",
        "unknown", "known", "unknown", "known", "unknown",
        "known", "unknown", "known", "unknown", "known",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "unknown", "known", None,
        "unknown", "known", "unknown", "known", "unknown",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "unknown", "known", "unknown",
        "known", "unknown", "known", "unknown", "known",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "unknown", "known", "unknown",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "unknown", "known", "unknown",
        "known", "unknown", "known", "unknown", "known",
        "unknown", "known", "unknown", "known", "unknown"
    ],
    "spam": [
        1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
})
x=df.drop(columns=['spam'])
y=df['spam']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5,random_state=45)
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, ["links", "words"]),
    ("cat", cat_pipe, ["sender"])
])
pipe=Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

param_grid = {
    "classifier__C": [0.1, 1, 10],
}
scoring={
     "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}
gs=GridSearchCV(pipe,param_grid,scoring=scoring,refit="f1",cv=3)
gs.fit(x_train,y_train) 
bm=gs.best_estimator_
val_prob = bm.predict_proba(x_val)[:, 1]
best_porog=0.5
best_f1=0
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                  0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                  0.8, 0.85, 0.9]:
    pred_val = [1 if prob >= threshold else 0 for prob in val_prob]
    f1 = f1_score(y_val, pred_val)
    if f1 > best_f1:
        best_f1 = f1
        best_porog = threshold
print("Лучший порог:", best_porog)
print("Лучший F1 на валидации:", best_f1)
test_prob = bm.predict_proba(x_test)[:, 1]
for threshold in [0.5, best_porog]:
    pred_test = [1 if prob >= threshold else 0 for prob in test_prob]

    print("Порог:", threshold)
    print("Precision:", precision_score(y_test, pred_test,))
    print("Recall:", recall_score(y_test, pred_test,))
    print("F1:", f1_score(y_test, pred_test))
    print("Accuracy:", accuracy_score(y_test, pred_test))