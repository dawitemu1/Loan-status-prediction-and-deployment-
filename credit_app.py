import joblib
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
# Assuming your logo is named "logo.png" and in the same directory
# logo_path = "logo.png"  # Adjust if your logo is in a different location
# Centered logo
# Load the Breast Cancer model
# Load the model and encoders
Loan_status_model = joblib.load(open('Loan_Status_xgb_model.sav',  'rb'))
label_encoders = joblib.load(open('label_encoders.pkl',  'rb'))
# Loan_XGB_model = joblib.load(open('C:/Users/Daveee/Desktop/CBE_project/Loan_Status_xgb_model_and_encoders.sav', 'rb'))
# # Loan_status_model = joblib.load(open('Loan_Status_xgb_model.sav',  'rb'))
# label_encoders = joblib.load(open('Loan_Status_label_encoders.sav',  'rb'))
# Load the label encoder
 # with open("C:/Users/Daveee/Desktop/Python apps/Iris_stremalit - Copy/label_encoders.pkl", "rb") as f:
 #    label_encoders = pickle.load(f)
st.image("logo.png", width=300)  # Change "logo.png" to the path of your logo image file
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Navigation_Menu',
                           [ 'Predicting Ioan_Status', 'EDA Loan Repayment Dynamics'],
                           icons=['bank', 'Ethiopia'],
                           default_index=0)

# Loan repayment status prediction and aproval limit
if selected == 'Predicting Ioan_Status':
    # page title
    # st.title('Predicting Loan Repayment & Recommend Approval Limit')
    st.markdown("<h3 style='text-align: center; color: black;'>Predicting Loan_status Using ML and Explanable AI </h3>", unsafe_allow_html=True)
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.selectbox("DISTRICTNAME", ["Head Office", "ARADA", "GONDER", "KIRKOS", "MEKELLE", "DIRE DAWA", "BAHIR DAR", "NEKEMTE", "HAWASSA", "MERKATO", "BOLE", "JIMMA", "SHIRE", "ADAMA", "METTU", "WOLAITA SODO","KALITI", "DEBRE MARKOS", "SHASHEMENE", "AMBO", "NIFAS SILK", "DESSIE", "HOSSANA", "MEGENAGNA", "GULLELLE", "DEBRE BREHAN", "YEKA", "JIJIGA", "KOLFE", "BALE ROBE", "ASSELA", "WOLDIA", "DILLA", "SEMERA"])

    with col2:
        Glucose = st.selectbox("REGIONNAME",  ["ADDIS ABABA", "OROMIYA", "AMHARA", "TIGRAY", "SIDAMA", "DIRE DAWA", "South Ethiopia", "SWER", "CENTAL ETHIOPIA", "BENSHANGUL GUMUZ", "SOMALE", "HARARI", "AFAR", "GAMBELLA"])

    with col3:
        BloodPressure = st.selectbox("CBE_REGION", ["CENTRAL", "NORTH EAST", "SOUTH WEST"])   

    with col1:
        SkinThickness = st.text_input('BRANCHNAME')
        
    with col2:
        Insulin = st.text_input('APPROVED_AMOUNT')

    with col3:
        BMI= st.selectbox("TENURE", ["Long Term", "Medium Term", "Short Term"])

    with col1:
        DiabetesPedigreeFunction = st.selectbox("TERM", ["SHORT TERM", "LONG TERM", "MEDIUM", "MEDIUM TERM"])

    with col2:
        Age = st.selectbox("LOAN_TYPE", ["TL", "OD", "CB", "AL", "IFB", "ML"])

    with col3:
        concavity_se = st.selectbox("LOAN_DESCRIPTION", ["Condominium Loan", "Working Capital & Project Loan", "Employee Salary Advance", "Coupon Bond", "Current Account with Overdraft", "Home Loan", "Adv Against Pre-Shipment Export LC", "Staff Mortgage Loan", "Partial Financing", "Consumer Durables Loan", "Agricultural Investment Term Loan", "Motor Vehicle Loan","Staff Personal Loan", "Construction Machinery Loan", "Murabaha - Corporate", "Agricultural Input Loan-Other","Micro-Finance Institution Loan", "Merchandise loan facility","CURRENT ACCOUNT", "Staff Automobile Loan", "Fertilizer  Loan", "Equipment/Machinery lease Financing", "Qardulhassen Financing", "Education Loan", "Advance Against Revolving Export LC", "Inter -Bank Lending", "Advance Against Import LC", "Personal Loan", "FCY Perssonal Loan", "Coffee Farm Term Loan Financing"])  
        
    with col1:
        concavepoints_se = st.selectbox("LOAN_PRODUCT", ["TERM.LOAN", "OD", "CONDO.COMM", "CONDO.2BED", "COUPON.BOND", "CONDO.3BED", "CONDO.1BED", "HOME.RES", "REV.PRESHIP", "STAFF.MORT", "CONDO.STDIO", "CONS.DURABLE", "AGRIC.INVEST.TERM", "PART.FIN.VEHIC", "CAR.LOAN", "STAFF.PERSONAL", "PART.FIN.BUILD", "LD", "CONSTR.MACH", "TERM.LOAN.CASHSEC", "AGRIC.LOAN.OTH", "MIC.FIN.INST", "NREV.PRESHIP", "LC.SETTLE", "AUTO.LOAN.NEW", "MERCH.OTHER", "QARD.AL.HASAN.FINANCE", "PART.FIN.MACH", "MERCH.NRV", "COFFEE.TERM.FIN", "EDU.LOAN", "SYND.LOAN","EQUIP.LESSOR", "FERT.LOAN", "GURANTEE.PRESHIP","INTER.BNK", "PERSONAL.LOAN.2W", "PERSONAL.USD"])

    with col2:
        texture_worst = st.selectbox('LTYPE', ["AA","OD", "LD"])

    with col3:
        perimeter_worst = st.text_input('CUST_SHORTNAME')
    with col1:
        area_worst = st.text_input('DAO_NAME')
       
    with col2:
        area_worst1 = st.text_input('PRINCIPAL_OS')
    with col3:
        area_worst2 = st.text_input('INTEREST_OS')
    with col1:
        area_worst3 = st.text_input('PRINCIPAL_ARREARS')
    with col2:
        area_worst4 = st.text_input('INTEREST_ARREARS')
    with col3:
        area_worst4 = st.text_input('CURRENT_COMMITTMENT')    
        
    with col1:
        area_worst5 = st.text_input('INSTALLMENT_AMOUNT')  
    with col2:
        area_worst6 = st.text_input('INSTALLMENT_FREQ_PRINCIPAL')   
       
    with col3:
        area_worst7 = st.text_input('INSTALLMENT_FREQ_INTEREST') 
        
    with col1:
        compactness_worst = st.selectbox("RISK_GRADE", ["RG3", "RG2", "RG1", "RG4", "RG5", "RG6", "RG8", "RG7"])
    with col2:
        compactness_worst1 = st.selectbox("ECONOMIC_SECTOR", ["Consumer and staff loans", "Domestic Trade", "Manufacturing", "Agriculture", "Hotel and Tourism", "Export", "Buliding  and Construction", "other sector", "Import", "Transport and Communication", "Health and Education", "Financial Institutions"])
    with col3:
        area_worst8 = st.text_input('INDUSTRY') 
    with col1:
        compactness_worst2 = st.selectbox("OWNERSHIP", ["PRIVATE", "GOVERNMENT", "PUBLIC", "COOPRATIVE"])
    with col2:
        compactness_worst3 = st.selectbox("SECTOR", ["Individual", "DOMESTIC TRADE", "GOVERNMENT SECTOR", "Manufacturing", "Export", "Agricultur hunt forestry & fishing", "Construction", "Hotel & tourism", "Import", "Electricity gas & water supply", "Transportation & storage", "Non Bank Fin intermed & insurance", "Banks", "CHURCH", "Health & soc work", "Real estate activities", "Education", "Profess scient & techn activities", "Act of hh gg & serv prodn for own", "Mining & quarrying", "Admin & support service activities", "Retail Small/Med Enterprise"])
    with col3:
        compactness_worst4 = st.selectbox("TERM_OF_PAYMENT", ["Monthly", "Quarterly", "Annually", "Half Yearly", "Bullrt (One lump sum upon Maturity)", "Daily"]) 
    with col1:
        compactness_worst5 = st.selectbox("PRODUCT_OWNER", ["Head Office", "MEKELLE", "GONDER", "ARADA", "DIRE DAWA", "NEKEMTE", "BAHIR DAR", "KIRKOS", "BUSINESS CREDIT", "Corporate Credit", "HAWASSA", "JIMMA", "SHIRE", "ADAMA", "WOLAITA SODO", "METTU", "MERKATO", "PUBLIC AND COOPERATIVE CREDIT", "LOAN WORKOUT", "DEBRE MARKOS", "SHASHEMENE", "BOLE", "KALITI", "AMBO", "DESSIE", "NIFAS SILK", "HOSSANA", "JIJIGA", "BALE ROBE", "MEGENAGNA","YEKA", "KOLFE", "GULLELLE", "WOLDIA", "DEBRE BREHAN", "ASSELA", "DILLA", "SEMERA","SMEs CREDIT"]) 
    with col2:
        area_worst9 = st.text_input('COLLATTERAL') 
    with col3:
        area_worst10 = st.text_input('COLLATERAL_VALUE') 
        
        
    # code for Prediction
    diab_diagnosis = ''
    # prediction_proba = ''

    # creating a button for Prediction
    if st.button('Loan_Status'):
        # Check if any field is empty
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,  DiabetesPedigreeFunction, Age, concavity_se, concavepoints_se, texture_worst, perimeter_worst, area_worst, area_worst1, area_worst2, area_worst3, area_worst4, area_worst5, area_worst6, area_worst7, compactness_worst, compactness_worst1, area_worst8, compactness_worst2, compactness_worst3, compactness_worst4, compactness_worst5, area_worst9, area_worst10]):
            st.error("Please fill all the fields.")
        else:
            # Perform prediction
            diab_prediction = Loan_status_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, concavity_se, concavepoints_se, texture_worst, perimeter_worst, area_worst, area_worst1, area_worst2, area_worst3, area_worst4, area_worst5, area_worst6, area_worst7, compactness_worst, compactness_worst1, area_worst8, compactness_worst2, compactness_worst3, compactness_worst4, compactness_worst5,  area_worst9, area_worst10]])

            # # Predict probabilities for each class
            # prediction_proba = Breastcancer_model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, concavity_se, concavepoints_se, texture_worst, perimeter_worst, area_worst, compactness_worst]])
            
            # Display prediction result
            if diab_prediction[0] == 0:
                diab_diagnosis = 'SET'
            elif diab_prediction[1] == 1:
                diab_diagnosis = 'PAS'
            elif diab_prediction[2] == 2:
                diab_diagnosis = 'SME'
            else:
                diab_diagnosis = 'NPL'
    
    st.success(diab_diagnosis)

#     # Display probability for each class
#     if isinstance(prediction_proba, np.ndarray) and prediction_proba.size > 0:
#         st.write('Probability for Customer Repayed sucessfully:', prediction_proba[0][1])
#         st.write('Probability for customer can not Repayed suessfully:', prediction_proba[0][0])

#         # Visualization of probability
#         probabilities = prediction_proba[0]
#         classes = ['Not_repayed', 'fully_Repayed']

#         fig, ax = plt.subplots()
#         ax.bar(classes, probabilities)
#         ax.set_ylabel('Probability')
#         ax.set_title('Probability Distribution')
#         st.pyplot(fig)

       
    
    
    
    
#     # EDA for CBE loan repayment system
# if selected == 'EDA Loan Repayment Dynamics':
#     # page title
#     # st.title('Exploring CBE Loan repayment system Dynamics in Ethiopia ')
#     st.markdown("<h4 style='text-align: center; color: black;'>Exploring CBE EDA for Loan Repayment status Dynamics in Ethiopia</h4>", unsafe_allow_html=True)
    
    
    
    
    
# st.markdown(
#     '`Create by` [Dawit Shibabaw](https://www.linkedin.com/in/dawit-shibabaw-3a0a98190/) | \
#          `Code:` [GitHub](https://github.com/dawitemu1)')







