import scikit_functions as sf
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


trv, trl = sf.piu_load_data()
X = sf.piu_prepare_values(trv)
y = sf.piu_prepare_labels(trl)

X_train, y_train, X_test, y_test = sf.piu_train_test_split(X, y)
 

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
 
y_pred = sgd_clf.predict(X_test)

sf.piu_print_classification_report(y_test, y_pred)

param_grid = [
    {'n_estimators': [3, 10, 30]},
    {'bootstrap': [False], 'n_estimators': [3, 10]},
  ]
 
rf_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_clf, param_grid, cv=5,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)
 
y_pred = grid_search.best_estimator_.predict(X_test)

sf.piu_print_classification_report(y_test, y_pred)

print(grid_search.best_estimator_)


# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix
#  
# print(cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="accuracy"))
#  
# y_train_pred = cross_val_predict(rf_clf, X_train, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
#  
# plt.matshow(conf_mx, cmap='gray') 
# plt.show()
