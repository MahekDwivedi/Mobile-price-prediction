
from sklearn.metrics import mean_absolute_error , mean_squared_error
print(" MEAN ABSOLUTE ERROR USING LINEAR REGRESSION\n")
print(mean_absolute_error( y_test , predictions))
np.sqrt( mean_squared_error( y_test , predictions))


print(" MEAN ABSOLUTE ERROR USING KNN ALGORITHM\n")
print(mean_absolute_error( y_test , knn_prediction))
np.sqrt( mean_squared_error( y_test , knn_prediction))


print(" MEAN ABSOLUTE ERROR USING RANDOM FOREST ALGORITHM\n")
print(mean_absolute_error( y_test , y_pred_rfc))
np.sqrt( mean_squared_error( y_test , y_pred_rfc))


print(" MEAN ABSOLUTE ERROR USING NAIVE BAYES ALGORITHM\n")
print(mean_absolute_error( y_test , y_pred_gnb))
np.sqrt( mean_squared_error( y_test , y_pred_gnb))
