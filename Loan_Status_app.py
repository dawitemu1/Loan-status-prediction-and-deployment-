import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="Loan_status Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.image("logo.jpg", width=400)  # Change "logo.png" to the path of your logo image file
# Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">Loan_status predict / credit healthy </h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #800080;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'
# Load the XGBoost model
model_path = "Loan_Status_xgb_model.sav"
loaded_model = pickle.load(open(model_path, "rb"))

# Load the label encoders
label_encoders_path = "label_encoders.pkl"
label_encoders = pickle.load(open(label_encoders_path, "rb"))

# Load the MinMax scalers
scalers_path = "minmax_scalers.pkl"
minmax_scalers = pickle.load(open(scalers_path, "rb"))

# Feature names and types
features = {
    'DISTRICTNAME': 'categorical',
    'REGIONNAME': 'categorical',
    'CBE_REGION': 'categorical',
    'BRANCHNAME': 'categorical',
    'APPROVED_AMOUNT': 'numerical',
    'TENURE': 'categorical',
    'TERM': 'categorical',
    'LOAN_TYPE': 'categorical',
    'LOAN_DESCRIPTION': 'categorical',
    'LOAN_PRODUCT': 'categorical',
    'LTYPE': 'categorical',
    'CUST_SHORTNAME': 'categorical',
    'DAO_NAME': 'categorical',
    'PRINCIPAL_OS': 'numerical',
    'INTEREST_OS': 'numerical',
    'PRINCIPAL_ARREARS': 'numerical',
    'INTEREST_ARREARS': 'numerical',
    'CURRENT_COMMITTMENT': 'numerical',
    'INSTALLMENT_AMOUNT': 'numerical',
    'INSTALLMENT_FREQ_PRINCIPAL': 'numerical',
    'INSTALLMENT_FREQ_INTEREST': 'numerical',
    'RISK_GRADE': 'categorical',
    'ECONOMIC_SECTOR': 'categorical', 
    'INDUSTRY': 'categorical',
    'OWNERSHIP': 'categorical',
    'SECTOR': 'categorical',
    'TERM_OF_PAYMENT': 'categorical',
    'PRODUCT_OWNER': 'categorical',
    'COLLATTERAL': 'categorical',
    'COLLATERAL_VALUE': 'numerical',
}
# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example XLSX input file](https://master/penguins_example.csv)
""")
# Create dictionary for grouping labels
group_labels = {
    'Bank_Adrress': ['DISTRICTNAME', 'REGIONNAME', 'CBE_REGION', 'BRANCHNAME'],
    'Credit_Information': ['APPROVED_AMOUNT', 'TENURE', 'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION',
       'LOAN_PRODUCT', 'LTYPE', 'CUST_SHORTNAME', 'DAO_NAME', 'PRINCIPAL_OS',
       'INTEREST_OS', 'PRINCIPAL_ARREARS', 'INTEREST_ARREARS',
       'CURRENT_COMMITTMENT', 'INSTALLMENT_AMOUNT',
       'INSTALLMENT_FREQ_PRINCIPAL', 'INSTALLMENT_FREQ_INTEREST', 'RISK_GRADE',
       'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 'SECTOR', 'TERM_OF_PAYMENT',
       'PRODUCT_OWNER'],
    'COLLATTERAL': ['COLLATTERAL', 'COLLATERAL_VALUE'],
    
}

# Option for CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])

# If CSV file is uploaded, read the file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# If CSV file is not uploaded, allow manual input
else:
    # Create empty dataframe to store input values
    input_df = pd.DataFrame(index=[0])

    # Loop through features and get user input
    # Loop through features and get user input
    for group, features_in_group in group_labels.items():
        st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
        st.sidebar.subheader(group)
        for feature in features_in_group:
            # Ensure each widget has a unique key
            widget_key = f"{group}_{feature}"

            # Display more descriptive labels
            if features[feature] == 'categorical':
                label = f"{feature.replace('_', ' ')}"
                input_df[feature] = st.sidebar.selectbox(label, label_encoders[feature].classes_, key=widget_key)
            else:
                label = f"{feature.replace('_', ' ')}"
                input_val = st.sidebar.text_input(label, key=widget_key)
                input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Display the input dataframe
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Loan_Status"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'LOAN_STATUS':
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))

    # Make predictions
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)
    prediction = loaded_model.predict(input_df)

    # Display the prediction
    st.sidebar.write("Prediction:", prediction[0])

    # Apply model to make predictions
    prediction_proba = loaded_model.predict_proba(input_df)

    st.subheader('Prediction (credit is healthy?)')
    LOAN_STATUS = np.array(['SET', 'PAS', 'SME', 'NPL'])
    st.write(LOAN_STATUS[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
