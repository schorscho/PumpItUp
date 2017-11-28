import piu_functions as piuf
import matplotlib.pyplot as plt


trv, trl = piuf.piu_load_data()
X = piuf.piu_prepare_values(trv)
y = piuf.piu_prepare_labels(trl)

X_train, y_train, X_test, y_test = piuf.piu_train_test_split(X, y)
 
from sklearn.linear_model import SGDClassifier
 
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
 
y_pred = sgd_clf.predict(X_test)

piuf.piu_print_classification_report(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10]},
  ]
 
rf_clf = RandomForestClassifier(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(rf_clf, param_grid, cv=5,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)
 
y_pred = grid_search.best_estimator_.predict(X_test)

piuf.piu_print_classification_report(y_test, y_pred)


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
