import joblib
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Loading the dataset
df = pd.read_csv("data/creditcard.csv")

#Show the first 5 rows of the dataset
print(df.head())

print("Dataset Shape:", df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nNull Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df['Class'].value_counts())

fraud_percentage = (
    df["Class"].value_counts()[1]
    /len(df)
)*100

print(f"\nFraud Percentage: {fraud_percentage:.4f}%")

# df["Class"].value_counts().plot(kind="bar")
# plt.title("Fraud vs Legitimate Transactions")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()

X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeature Matrix Shape:", X.shape)

print("Target Vector Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)
model = LogisticRegression(max_iter=1000)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Complete!")

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")

print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: ")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "ml/model.pkl")

print("\nModel saved successfully!")
