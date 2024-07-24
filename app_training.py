import joblib
import streamlit as st
from streamlit_option_menu import option_menu

# Load the model and encoders
Loan_status_model = joblib.load(open('Loan_Status_xgb_model.sav',  'rb'))
label_encoders = joblib.load(open('label_encoders.pkl',  'rb'))

# Centered logo
st.image ("logo.png", width=300)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Navigation_Menu',
                           ['Predicting Loan_Status', 'EDA Loan Repayment Dynamics'],
                           icons=['bank', 'Ethiopia'],
                           default_index=0)

# Loan repayment status prediction and approval limit
if selected == 'Predicting Loan_Status':
    st.markdown("<h3 style='text-align: center; color: black;'>Predicting Loan_status Using ML and Explainable AI </h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        DISTRICTNAME = st.selectbox("DISTRICTNAME", ["Head Office", "ARADA", "GONDER", "KIRKOS", "MEKELLE", "DIRE DAWA", "BAHIR DAR", "NEKEMTE", "HAWASSA", "MERKATO", "BOLE", "JIMMA", "SHIRE", "ADAMA", "METTU", "WOLAITA SODO", "KALITI", "DEBRE MARKOS", "SHASHEMENE", "AMBO", "NIFAS SILK", "DESSIE", "HOSSANA", "MEGENAGNA", "GULLELLE", "DEBRE BREHAN", "YEKA", "JIJIGA", "KOLFE", "BALE ROBE", "ASSELA", "WOLDIA", "DILLA", "SEMERA"])
    with col2:
        REGIONNAME = st.selectbox("REGIONNAME", ["ADDIS ABABA", "OROMIYA", "AMHARA", "TIGRAY", "SIDAMA", "DIRE DAWA", "South Ethiopia", "SWER", "CENTAL ETHIOPIA", "BENSHANGUL GUMUZ", "SOMALE", "HARARI", "AFAR", "GAMBELLA"])
    with col3:
        CBE_REGION = st.selectbox("CBE_REGION", ["CENTRAL", "NORTH EAST", "SOUTH WEST"])

    with col1:
        BRANCHNAME = st.text_input('BRANCHNAME')
    with col2:
        APPROVED_AMOUNT = st.text_input('APPROVED_AMOUNT')
    with col3:
        TENURE = st.selectbox("TENURE", ["Long Term", "Medium Term", "Short Term"])

    with col1:
        TERM = st.selectbox("TERM", ["SHORT TERM", "LONG TERM", "MEDIUM", "MEDIUM TERM"])
    with col2:
        LOAN_TYPE = st.selectbox("LOAN_TYPE", ["TL", "OD", "CB", "AL", "IFB", "ML"])
    with col3:
        LOAN_DESCRIPTION = st.selectbox("LOAN_DESCRIPTION", ["Condominium Loan", "Working Capital & Project Loan", "Employee Salary Advance", "Coupon Bond", "Current Account with Overdraft", "Home Loan", "Adv Against Pre-Shipment Export LC", "Staff Mortgage Loan", "Partial Financing", "Consumer Durables Loan", "Agricultural Investment Term Loan", "Motor Vehicle Loan", "Staff Personal Loan", "Construction Machinery Loan", "Murabaha - Corporate", "Agricultural Input Loan-Other", "Micro-Finance Institution Loan", "Merchandise loan facility", "CURRENT ACCOUNT", "Staff Automobile Loan", "Fertilizer  Loan", "Equipment/Machinery lease Financing", "Qardulhassen Financing", "Education Loan", "Advance Against Revolving Export LC", "Inter-Bank Lending", "Advance Against Import LC", "Personal Loan", "FCY Personal Loan", "Coffee Farm Term Loan Financing"])

    with col1:
        LOAN_PRODUCT = st.selectbox("LOAN_PRODUCT", ["TERM.LOAN", "OD", "CONDO.COMM", "CONDO.2BED", "COUPON.BOND", "CONDO.3BED", "CONDO.1BED", "HOME.RES", "REV.PRESHIP", "STAFF.MORT", "CONDO.STDIO", "CONS.DURABLE", "AGRIC.INVEST.TERM", "PART.FIN.VEHIC", "CAR.LOAN", "STAFF.PERSONAL", "PART.FIN.BUILD", "LD", "CONSTR.MACH", "TERM.LOAN.CASHSEC", "AGRIC.LOAN.OTH", "MIC.FIN.INST", "NREV.PRESHIP", "LC.SETTLE", "AUTO.LOAN.NEW", "MERCH.OTHER", "QARD.AL.HASAN.FINANCE", "PART.FIN.MACH", "MERCH.NRV", "COFFEE.TERM.FIN", "EDU.LOAN", "SYND.LOAN", "EQUIP.LESSOR", "FERT.LOAN", "GURANTEE.PRESHIP", "INTER.BNK", "PERSONAL.LOAN.2W", "PERSONAL.USD"])
    with col2:
        LTYPE = st.selectbox('LTYPE', ["AA", "OD", "LD"])
    with col3:
        CUST_SHORTNAME = st.text_input('CUST_SHORTNAME')

    with col1:
        DAO_NAME = st.text_input('DAO_NAME')
    with col2:
        PRINCIPAL_OS = st.text_input('PRINCIPAL_OS')
    with col3:
        INTEREST_OS = st.text_input('INTEREST_OS')

    with col1:
        PRINCIPAL_ARREARS = st.text_input('PRINCIPAL_ARREARS')
    with col2:
        INTEREST_ARREARS = st.text_input('INTEREST_ARREARS')
    with col3:
        CURRENT_COMMITTMENT = st.text_input('CURRENT_COMMITTMENT')

    with col1:
        INSTALLMENT_AMOUNT = st.text_input('INSTALLMENT_AMOUNT')
    with col2:
        INSTALLMENT_FREQ_PRINCIPAL = st.text_input('INSTALLMENT_FREQ_PRINCIPAL')
    with col3:
        INSTALLMENT_FREQ_INTEREST = st.text_input('INSTALLMENT_FREQ_INTEREST')

    with col1:
        RISK_GRADE = st.selectbox("RISK_GRADE", ["RG3", "RG2", "RG1", "RG4", "RG5", "RG6", "RG8", "RG7"])
    with col2:
        ECONOMIC_SECTOR = st.selectbox("ECONOMIC_SECTOR",  ["Consumer and staff loans", "Domestic Trade", "Manufacturing", "Agriculture", "Hotel and Tourism", "Export", "Buliding  and Construction", "other sector", "Import", "Transport and Communication", "Health and Education", "Financial Institutions"])
    with col3:
        INDUSTRY = st.text_input('INDUSTRY')
    with col1:
        OWNERSHIP = st.selectbox("OWNERSHIP", ["PRIVATE", "GOVERNMENT", "PUBLIC", "COOPRATIVE"])
    with col2:
        SECTOR = st.selectbox("SECTOR", ["Individual", "DOMESTIC TRADE", "GOVERNMENT SECTOR", "Manufacturing", "Export", "Agricultur hunt forestry & fishing", "Construction", "Hotel & tourism", "Import", "Electricity gas & water supply", "Transportation & storage", "Non Bank Fin intermed & insurance", "Banks", "CHURCH", "Health & soc work", "Real estate activities", "Education", "Profess scient & techn activities", "Act of hh gg & serv prodn for own", "Mining & quarrying", "Admin & support service activities", "Retail Small/Med Enterprise"])
    with col3:
        TERM_OF_PAYMENT = st.selectbox("TERM_OF_PAYMENT", ["Monthly", "Quarterly", "Annually", "Half Yearly", "Bullrt (One lump sum upon Maturity)", "Daily"]) 

    with col1:
        PRODUCT_OWNER =  st.selectbox("PRODUCT_OWNER", ["Head Office", "MEKELLE", "GONDER", "ARADA", "DIRE DAWA", "NEKEMTE", "BAHIR DAR", "KIRKOS", "BUSINESS CREDIT", "Corporate Credit", "HAWASSA", "JIMMA", "SHIRE", "ADAMA", "WOLAITA SODO", "METTU", "MERKATO", "PUBLIC AND COOPERATIVE CREDIT", "LOAN WORKOUT", "DEBRE MARKOS", "SHASHEMENE", "BOLE", "KALITI", "AMBO", "DESSIE", "NIFAS SILK", "HOSSANA", "DEBRE BREHAN", "JIJIGA", "BALE ROBE", "MEGENAGNA", "YEKA", "KOLFE", "GULLELLE", "WOLDIA", "ASSELA", "DILLA", "SEMERA","SMEs CREDIT"]) 
    with col2:
        COLLATERAL = st.text_input('COLLATERAL')
    with col3:
        COLLATERAL_VALUE = st.text_input('COLLATERAL_VALUE')
        
    # code for Prediction
    diab_diagnosis =''
 # creating a button for Prediction
    if st.button('Loan_Status'):
        # Check if any field is empty
        if not all([DISTRICTNAME, REGIONNAME, CBE_REGION, BRANCHNAME, APPROVED_AMOUNT, TENURE, TERM, LOAN_TYPE, LOAN_DESCRIPTION, LOAN_PRODUCT, LTYPE, CUST_SHORTNAME, DAO_NAME, PRINCIPAL_OS, INTEREST_OS, PRINCIPAL_ARREARS, INTEREST_ARREARS, CURRENT_COMMITTMENT, INSTALLMENT_AMOUNT, INSTALLMENT_FREQ_PRINCIPAL, INSTALLMENT_FREQ_INTEREST, RISK_GRADE, ECONOMIC_SECTOR, INDUSTRY, OWNERSHIP, SECTOR, TERM_OF_PAYMENT, PRODUCT_OWNER, COLLATERAL, COLLATERAL_VALUE]):
            st.error("Please fill all the fields.")
        else:
            # Prepare features for prediction
            features = [DISTRICTNAME, REGIONNAME, CBE_REGION, BRANCHNAME, APPROVED_AMOUNT, TENURE, TERM, LOAN_TYPE, LOAN_DESCRIPTION, LOAN_PRODUCT, LTYPE, CUST_SHORTNAME, DAO_NAME, PRINCIPAL_OS, INTEREST_OS, PRINCIPAL_ARREARS, INTEREST_ARREARS, CURRENT_COMMITTMENT, INSTALLMENT_AMOUNT, INSTALLMENT_FREQ_PRINCIPAL, INSTALLMENT_FREQ_INTEREST, RISK_GRADE, ECONOMIC_SECTOR, INDUSTRY, OWNERSHIP, SECTOR, TERM_OF_PAYMENT, PRODUCT_OWNER, COLLATERAL, COLLATERAL_VALUE]

            # Encode categorical features if necessary
            for i, feature in enumerate(features):
                if feature in label_encoders:
                    features[i] = label_encoders[feature].transform([features[i]])[0]

            # Convert features to the appropriate format
            features = [features]

            # Perform prediction
            diab_prediction = Loan_status_model.predict(features)

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
