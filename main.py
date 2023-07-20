from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv("train.csv")

df.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare'], axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

feature_columns = ['PassengerId', 'Pclass', 'SibSp', 'Parch',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

target = "Survived"

X_train = df[feature_columns]
y_train = df[target]

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

df = pd.read_csv("test.csv")

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df.dropna(axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

pred = model.predict(df)

df = pd.DataFrame({"PassengerId":list(df['PassengerId']), "Survived":list(pred)})

df.to_csv("submission.csv", index=False)