import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
X = df.iloc[:, :-1]   # All columns except last
y = df.iloc[:, -1]    # Last column (species)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:",confusion_matrix(y_test, y_pred))
print("\nClassification Report:",classification_report(y_test, y_pred))
