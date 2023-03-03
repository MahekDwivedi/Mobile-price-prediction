from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def my_confusion_matrix(y_test, y_pred, plt_title):
    cm=confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return cm
  
  
  
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(bootstrap= True,
                           max_depth= 7,
                           max_features= 15,
                           min_samples_leaf= 3,
                           min_samples_split= 10,
                           n_estimators= 200,
                           random_state=7)


rfc.fit(x_train, y_train)
y_pred_rfc=rfc.predict(x_test)


print('Random Forest Classifier Accuracy Score: ',accuracy_score(y_test ,y_pred_rfc))
cm_rfc=my_confusion_matrix(y_test, y_pred_rfc, 'Random Forest Confusion Matrix')


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, random_y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, random_y_pred) 
print("Accuracy score:",accuracy)

