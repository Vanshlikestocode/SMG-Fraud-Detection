from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("ml/model.pkl")


class Transaction(BaseModel): 
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float





@app.get("/")
def home():
    return {
        "project": "SMG Fraud Detection Platform",
        "status": "running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    features = [[
    transaction.Time,
    transaction.V1,
    transaction.V2,
    transaction.V3,
    transaction.V4,
    transaction.V5,
    transaction.V6,
    transaction.V7,
    transaction.V8,
    transaction.V9,
    transaction.V10,
    transaction.V11,
    transaction.V12,
    transaction.V13,
    transaction.V14,
    transaction.V15,
    transaction.V16,
    transaction.V17,
    transaction.V18,
    transaction.V19,
    transaction.V20,
    transaction.V21,
    transaction.V22,
    transaction.V23,
    transaction.V24,
    transaction.V25,
    transaction.V26,
    transaction.V27,
    transaction.V28,
    transaction.Amount
    ]]

    try:
        prediction = model.predict(features)[0]

        fraud_probability = model.predict_proba(features)[0][1]
        
        return {
        "status": "success",
        "prediction": int(prediction),
        "transaction_type":
            "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability":
            round(float(fraud_probability), 4)
    }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
