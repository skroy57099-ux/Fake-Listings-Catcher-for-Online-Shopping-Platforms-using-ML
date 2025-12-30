📌 Project Conclusion

This project implemented a Fake Listings Detection System using structured e-commerce metadata.  
The dataset contained listing-level behavioral and descriptive features that strongly correlate with counterfeit activity.

🔍 Key Insights
- Short and low-effort descriptions are reliable fraud indicators
- Sellers with few reviews and lower ratings frequently list counterfeit items
- Longer shipping times and incomplete contact/return info increase the likelihood of fraud
- Spelling errors were a surprisingly strong signal of fake listings

🧪 Model Performance
All three models (Logistic Regression, RandomForest, XGBoost) achieved perfect classification metrics:
- Accuracy: **100%**
- Precision: **100%**
- Recall: **100%**
- F1-score: **100%**

This result is explained by the highly discriminative metadata contained in the dataset.  
The decision boundary between genuine and counterfeit listings is extremely clear.

📝 Model Selection
- Logistic Regression used as baseline
- RandomForest chosen as final model due to interpretability and feature importance insights
- XGBoost verified stability of performance across boosting methods

⚙️ Deployment Readiness
The RandomForest model and preprocessing scaler were saved using Joblib.  
Inference is demonstrated using a single-row prediction example.

🚧 Limitations
- Dataset is clean and strongly predictive; real-world data would contain noise
- Image and text fields were not fully utilized in this version
- Further testing needed on scraped or user-generated listing data

🚀 Future Enhancements
- Collect real-world listing data to evaluate generalization
- Integrate NLP models to score description quality
- Add image deduplication and reverse image search for visual fraud detection
- Deploy model as REST API or Streamlit app



# How to run this project:
# usage example
import joblib
model = joblib.load("models/fake_listings_rf_model.pkl")
scaler = joblib.load("models/fake_listings_scaler.pkl")
