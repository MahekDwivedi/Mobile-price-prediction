# Linear Regression Algorithm

from sklearn.linear_model import LinearRegression
model= LinearRegression().fit( x_train , y_train )
predictions = model.predict(x_test)

print(df["price_range"].mean() )

test_residual = y_test - predictions
test_residual 
