import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
param_grid = {"criterion": ["gini", "entropy"],"max_depth": [2, 3, 4, 5, None],"min_samples_split": [2, 4, 6]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid=param_grid,cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")