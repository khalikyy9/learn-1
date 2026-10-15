import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.DataFrame({
    "links_count": [
        0, 1, 8, 2, 5,
        2, 4, 1, 0, 3,
        3, 0, 6, 2, 1,
        5, 1, 4, 0, 7
    ],
    "words_count": [
        120, 250, 180, 90, 95,
        300, 140, 220, 150, 75,
        110, 280, 190, 200, 170,
        160, 240, 100, 310, 85
    ],
    "suspicious_words": [
        0, 2, 3, 1, 4,
        1, 2, 0, 0, 3,
        2, 0, 4, 1, 2,
        2, 1, 3, 0, 5
    ],
    "sender_known": [
        1, 1, 0, 1, 0,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 1,
        0, 1, 0, 1, 0
    ],
    "spam": [
        0, 1, 0, 0, 1,
        1, 0, 1, 1, 0,
        0, 1, 0, 0, 1,
        1, 1, 0, 0, 1
    ]
})
print('исходный датасет:')
print(df)
x=df.drop('spam',axis=1)
y=df['spam']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("\nКраткий вывод:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")


