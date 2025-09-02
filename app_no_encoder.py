import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load model, features, and scaler
with open("model_no_encoder.pkl", "rb") as f:
    model, feature_columns, scaler = pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="💼 Salary Prediction (Advanced)", layout="centered")
st.title("💼 Salary Prediction App (Advanced)")

def user_input_features():
    data = {
        'age': st.number_input("Age", 17, 90, 30),
        'workclass': st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                                'Federal-gov', 'Local-gov', 'State-gov',
                                                'Without-pay', 'Never-worked']),
        'fnlwgt': st.number_input("Fnlwgt", 10000, 1000000, 150000),
        'education': st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                                                'Some-college', 'Assoc-acdm', 'Assoc-voc',
                                                'Doctorate', 'Prof-school', 'Preschool',
                                                '10th', '12th', '1st-4th', '5th-6th', '7th-8th']),
        'educational-num': st.slider("Education Number", 1, 16, 10),
        'marital-status': st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced',
                                                          'Never-married', 'Separated',
                                                          'Widowed', 'Married-spouse-absent']),
        'occupation': st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service',
                                                  'Sales', 'Exec-managerial', 'Prof-specialty',
                                                  'Handlers-cleaners', 'Machine-op-inspct',
                                                  'Adm-clerical', 'Farming-fishing',
                                                  'Transport-moving', 'Priv-house-serv',
                                                  'Protective-serv', 'Armed-Forces']),
        'relationship': st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband',
                                                      'Not-in-family', 'Other-relative',
                                                      'Unmarried']),
        'race': st.selectbox("Race", ['White', 'Asian-Pac-Islander',
                                      'Amer-Indian-Eskimo', 'Other', 'Black']),
        'gender': st.selectbox("Gender", ['Female', 'Male']),
        'capital-gain': st.number_input("Capital Gain", min_value=0, value=0),
        'capital-loss': st.number_input("Capital Loss", min_value=0, value=0),
        'hours-per-week': st.slider("Hours per Week", 1, 100, 40),
        'native-country': st.selectbox("Native Country", ['United-States', 'India', 'Mexico',
                                                          'Philippines', 'Germany', 'Canada',
                                                          'England', 'China', 'Cuba', 'Iran',
                                                          'South', 'Vietnam', 'Puerto-Rico'])
    }
    return pd.DataFrame([data])

# --- User Input ---
df_input = user_input_features()

# --- Predict & Visualize ---
if st.button("🔮 Predict Income", key="predict"):
    try:
        # One-hot encode and align
        df_encoded = pd.get_dummies(df_input)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
        df_scaled = scaler.transform(df_encoded)
        
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)

        st.success(f"✅ Predicted Income: {prediction[0]}")
        st.info(f"🧮 Probability (>50K): {probability[0][1]:.2%}")
        st.metric("Confidence Level", f"{probability[0][1]*100:.2f}%")

        # --- Feature Importance ---
        st.subheader("📌 Top 10 Most Influential Features")
        feature_importance = pd.Series(np.abs(model.coef_[0]), index=feature_columns)
        top_features = feature_importance.sort_values(ascending=False).head(10)
        st.bar_chart(top_features)

             # SHAP Explainability (Local + Global)
        st.subheader("💡 SHAP Local Explanation")

        # Prepare DataFrame for SHAP input
        df_for_shap = pd.DataFrame(df_scaled, columns=feature_columns)

        import shap
        explainer = shap.Explainer(model, df_for_shap)
        shap_values = explainer(df_for_shap)

                # SHAP Waterfall Plot (Local)
        st.subheader("🔍 SHAP Waterfall Plot (Single Prediction)")
        fig_local = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_local.figure)

        # SHAP Summary Plot (Global)
        st.subheader("📊 SHAP Summary Plot (Global Feature Impact)")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, df_for_shap, show=False)
        st.pyplot(fig_summary)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")



        import pickle
import pandas as pd

# Load model
with open("model_no_encoder.pkl", "rb") as f:
    model, feature_columns, scaler = pickle.load(f)

# Test data
test = pd.DataFrame([{
    'age': 35,
    'workclass': 'Private',
    'fnlwgt': 150000,
    'education': 'Bachelors',
    'educational-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Prof-specialty',
    'relationship': 'Not-in-family',
    'race': 'White',
    'gender': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}])

# Preprocess
df_encoded = pd.get_dummies(test)
df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
df_scaled = scaler.transform(df_encoded)

# Predict
print(model.predict(df_scaled))



import pandas as pd
import pickle

# Load model
with open("model_no_encoder.pkl", "rb") as f:
    model, feature_columns, scaler = pickle.load(f)

# Create a list of test users
test_users = [
    {'age': 28, 'workclass': 'Private', 'fnlwgt': 150000, 'education': 'Bachelors',
     'educational-num': 13, 'marital-status': 'Never-married', 'occupation': 'Sales',
     'relationship': 'Not-in-family', 'race': 'White', 'gender': 'Male',
     'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'},
    
    {'age': 45, 'workclass': 'Self-emp-inc', 'fnlwgt': 180000, 'education': 'Masters',
     'educational-num': 15, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
     'relationship': 'Husband', 'race': 'White', 'gender': 'Male',
     'capital-gain': 10000, 'capital-loss': 0, 'hours-per-week': 50, 'native-country': 'Canada'},

    {'age': 22, 'workclass': 'Private', 'fnlwgt': 120000, 'education': 'HS-grad',
     'educational-num': 9, 'marital-status': 'Never-married', 'occupation': 'Adm-clerical',
     'relationship': 'Own-child', 'race': 'Black', 'gender': 'Female',
     'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 30, 'native-country': 'Mexico'}
]

df = pd.DataFrame(test_users)

# Encode, align, scale
df_encoded = pd.get_dummies(df)
df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
df_scaled = scaler.transform(df_encoded)

# Predict
predictions = model.predict(df_scaled)
probabilities = model.predict_proba(df_scaled)

# Show results
for i, user in enumerate(test_users):
    print(f"\nUser {i+1}:")
    print(f"Data: {user}")
    print(f"Prediction: {predictions[i]}")
    print(f"Probability (>50K): {probabilities[i][1]:.2%}")
