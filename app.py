import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
from langchain_google_genai import ChatGoogleGenerativeAI

# ===========================
# CONFIGURATION
# ===========================
app = Flask(__name__)
MODEL_PATH = "best_xgb_model.pkl"

# ✅ Hardcoded Gemini API key
GOOGLE_API_KEY = "AIzaSyApqYEwauS3jO8o_dsG17Kk2JoDKsCnZkg"

# Load trained model
model = joblib.load(MODEL_PATH)

# Initialize Gemini LLM (use currently supported model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_output_tokens=1024
)

# ===========================
# Routes
# ===========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Collect raw inputs
        input_data = {
            "Administrative": float(request.form.get("Administrative", 0)),
            "Informational": float(request.form.get("Informational", 0)),
            "ProductRelated": float(request.form.get("ProductRelated", 0)),
            "BounceRates": float(request.form.get("BounceRates", 0)),
            "ExitRates": float(request.form.get("ExitRates", 0)),
            "PageValues": float(request.form.get("PageValues", 0)),
            "SpecialDay": float(request.form.get("SpecialDay", 0)),
            "Month": request.form.get("Month", "May"),
            "OperatingSystems": int(request.form.get("OperatingSystems", 1)),
            "Browser": int(request.form.get("Browser", 1)),
            "Region": int(request.form.get("Region", 1)),
            "TrafficType": int(request.form.get("TrafficType", 1)),
            "VisitorType": request.form.get("VisitorType", "Returning_Visitor"),
            "Weekend": int(request.form.get("Weekend", 0)),
            "ProductRelated_Duration": float(request.form.get("ProductRelated_Duration", 0)),
            "Administrative_Duration": float(request.form.get("Administrative_Duration", 0)),
            "Informational_Duration": float(request.form.get("Informational_Duration", 0)),
        }

        # 2️⃣ Feature Engineering
        Total_Page_Events = input_data["Administrative"] + input_data["Informational"] + input_data["ProductRelated"]
        Total_Duration = input_data["Administrative_Duration"] + input_data["Informational_Duration"] + input_data["ProductRelated_Duration"]
        Avg_Page_Duration = Total_Duration / Total_Page_Events if Total_Page_Events > 0 else 0
        Pages_per_Session = Total_Page_Events
        Product_Time_x_Count = input_data["ProductRelated"] * input_data["ProductRelated_Duration"]

        def safe_log(x): return np.log1p(max(x,0))

        df = pd.DataFrame([{
            **input_data,
            "Total_Page_Events": Total_Page_Events,
            "Total_Duration": Total_Duration,
            "Avg_Page_Duration": Avg_Page_Duration,
            "Pages_per_Session": Pages_per_Session,
            "Product_Time_x_Count": Product_Time_x_Count,
            "Pages_per_Session_log": safe_log(Pages_per_Session),
            "ProductRelated_log": safe_log(input_data["ProductRelated"]),
            "BounceRates_log": safe_log(input_data["BounceRates"]),
            "ExitRates_log": safe_log(input_data["ExitRates"]),
            "PageValues_log": safe_log(input_data["PageValues"]),
            "Informational_log": safe_log(input_data["Informational"]),
            "Administrative_log": safe_log(input_data["Administrative"]),
            "Total_Duration_log": safe_log(Total_Duration),
        }])

        # 3️⃣ Make Prediction
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        # 4️⃣ Gemini Insights
        prompt = f"""
        You are an e-commerce analytics expert.
        A session with the following values was analyzed:

        {df.to_string(index=False)}

        Model prediction: {'✅ Customer is likely to purchase.' if pred == 1 else '❌ Customer is unlikely to purchase.'}
        Probability: {round(proba*100,2)}%

        Please explain in simple terms:
        - What this prediction means
        - Why the model might have made this decision
        - Two business suggestions to increase purchases
        """

        try:
            insight = llm.invoke(prompt)
        except Exception as e:
            insight = f"LLM failed to generate insight: {e}"

        return render_template(
            "result.html",
            input_data=input_data,
            prediction="✅ Will Purchase" if pred == 1 else "❌ Will NOT Purchase",
            probability=round(proba * 100, 2),
            insight=insight
        )
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
