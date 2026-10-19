import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.DataFrame({
    "area": [
        30, 45, 60, 75, 90,
        55, 40, None, 80, 65,
        35, 100, 50, 70, 85,
        42, 95, 58, 78, 62,
    ],
    "rooms": [
        1, 2, 2, 3, 3,
        2, 1, 2, 3, 2,
        1, 4, 2, None, 3,
        1, 4, 2, 3, 2,
    ],
    "price": [
        5_000_000, 7_000_000, 9_000_000, 11_000_000, 13_000_000,
        8_000_000, 5_500_000, 8_500_000, 12_000_000, 8_500_000,
        4_800_000, 15_000_000, 7_500_000, 10_000_000, 12_500_000,
        5_800_000, 14_000_000, 8_200_000, 11_500_000, 7_000_000,
    ],
})
x = df.drop(columns=["price"])
y = df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)
imputer = SimpleImputer(strategy="median")
x_train=imputer.fit_transform(x_train)
x_test=imputer.transform(x_test)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LinearRegression().fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mse**0.5
r2=r2_score(y_test,y_pred)
print('mse:',mse)
print('mae:',mae)
print('rmse:',rmse)
print('r2:',r2)
print(imputer.statistics_)