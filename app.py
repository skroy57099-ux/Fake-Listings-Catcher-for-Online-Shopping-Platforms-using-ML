import joblib
import gradio as gr
import pandas as pd

# load model + scaler
model = joblib.load("models/fake_listings_rf_model.pkl")
scaler = joblib.load("models/fake_listings_scaler.pkl")

# define features HuggingFace will receive from the UI
feature_columns = [
    "price", "seller_rating", "seller_reviews", "purchases", 
    "views", "description_length", "shipping_time_days", 
    "contact_info_complete", "return_policy_clear", 
    "certification_badges", "warranty_months", "spelling_errors"
]

def predict_listing(*values):
    df = pd.DataFrame([values], columns=feature_columns)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]
    label = "FAKE" if pred == 1 else "GENUINE"
    return f"{label} (Confidence: {proba:.2f})"

inputs = [gr.Number(label=col) for col in feature_columns]
output = gr.Textbox(label="Prediction Result")

demo = gr.Interface(
    fn=predict_listing, 
    inputs=inputs, 
    outputs=output, 
    title="Fake Listing Detector",
    description="Upload listing details to detect if it's fake or genuine"
)

if __name__ == "__main__":
    demo.launch()