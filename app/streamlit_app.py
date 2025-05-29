import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

# Load model and features
model = joblib.load("../models/tuned/best_random_forest.pkl")
feature_list = joblib.load("../models/tuned/features.pkl")

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 House Price Prediction App")
st.markdown("Estimate a house's sale price by filling in key features in the sidebar.")

# Sidebar inputs
st.sidebar.header("📋 House Features")

# Collect inputs
input_values = {
    "GrLivArea": st.sidebar.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500),
    "GarageArea": st.sidebar.number_input("Garage Area (sqft)", 0, 1500, 400),
    "TotalBsmtSF": st.sidebar.number_input("Total Basement Area (sqft)", 0, 3000, 800),
    "YearBuilt": st.sidebar.slider("Year Built", 1900, 2025, 2000),
    "FullBath": st.sidebar.selectbox("Full Bathrooms", [0, 1, 2, 3]),
    "Fireplaces": st.sidebar.selectbox("Fireplaces", [0, 1, 2, 3]),
    "LotArea": st.sidebar.number_input("Lot Area (sqft)", 1000, 200000, 10000),
    "OverallQual": st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5),
    "1stFlrSF": st.sidebar.number_input("1st Floor Area (sqft)", 300, 3000, 800),
    "2ndFlrSF": st.sidebar.number_input("2nd Floor Area (sqft)", 0, 2000, 400),
    "BedroomAbvGr": st.sidebar.selectbox("Bedrooms Above Ground", [0, 1, 2, 3, 4, 5]),
    "KitchenAbvGr": st.sidebar.selectbox("Kitchens Above Ground", [0, 1, 2]),
    "MSZoning_RL": st.sidebar.selectbox("Zoning RL", [0, 1])  # example one-hot column
}

# Fill all features for model input
full_input = {}
for feature in feature_list:
    full_input[feature] = input_values.get(feature, 0)

input_df = pd.DataFrame([full_input])

# Show input summary
st.subheader("📊 Input Summary")
st.write(input_df.T.rename(columns={0: 'Value'}))

# Predict
if st.button("🔮 Predict Sale Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.0f}")

    # Download as CSV
    input_df["PredictedPrice"] = prediction
    csv_buffer = io.StringIO()
    input_df.to_csv(csv_buffer, index=False)
    st.download_button("📄 Download Prediction CSV", data=csv_buffer.getvalue(),
                       file_name="house_price_prediction.csv", mime="text/csv")

    # Feature importances
    st.subheader("📌 Top Feature Importances")
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[-10:][::-1]
    top_features = [feature_list[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]

    fig, ax = plt.subplots()
    ax.barh(top_features[::-1], top_importances[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")
    st.pyplot(fig)

# Sidebar: Model evaluation info
st.sidebar.markdown("---")
st.sidebar.markdown("📈 **Model Performance**")
st.sidebar.write("R² Score: `0.87`")
st.sidebar.write("RMSE: `~$24,000`")
st.sidebar.markdown("This model is trained on a dataset of house sales and predicts the sale price based on various features.")
# Footer    
st.sidebar.markdown("---")
st.sidebar.markdown("🏠 **Developed by**: Ayrin Akter Supty")