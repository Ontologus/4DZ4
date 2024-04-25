import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score

df = pd.read_csv('/content/titanic.csv')
df.drop(['Cabin', 'Embarked', 'Ticket', 'Gender', 'Name', 'PassengerId'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(value=df['Age'].mean())

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def f1(precision, recall):
  return (2 * precision * recall) / (precision + recall)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
f1_knn = f1(precision_score(y_test, y_pred_knn), recall_score(y_test, y_pred_knn))
print(f1_knn, '- knn')

logreg = LogisticRegression(max_iter=450)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
f1_logreg = f1(precision_score(y_test, y_pred_logreg), recall_score(y_test, y_pred_logreg))
print(f1_logreg, '- logreg')

#f1 больше то у knn, то у logreg