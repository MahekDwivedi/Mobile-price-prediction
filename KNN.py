# data preprocessing

x=df.drop(['price_range'],axis=1)
y=df['price_range']

from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=0)


#KNN classification
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train, y_train)

knn.score(x_train, y_train)
knn_prediction= knn.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn_prediction)

print('KNN Classifier Accuracy Score: ',accuracy_score(y_test ,knn_prediction))
cm_rfc=my_confusion_matrix(y_test, knn_prediction, 'KNN Confusion Matrix')
