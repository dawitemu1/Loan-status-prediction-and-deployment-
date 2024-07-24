import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import calendar  # To get month names

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="EDA for Loan status",
    page_icon="/home/U5MRIcon.png",
    layout="wide",
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
st.markdown('<h1 style="text-align: center;">Explore Loan EDA Dynamics CBE</h1>', unsafe_allow_html=True)

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

# Defining header color and font
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


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)


@st.cache_data
def plot_categorical_feature(data, categorical_feature, figsize1=(8, 5), figsize2=(6, 6), figsize3=(12, 8)):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")
    # Convert the 'edhs_year' column to string
    data['year'] = data['year'].astype(str)

    try:
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(figsize1[0] + figsize2[0], max([figsize1[1], figsize2[1]])))
        gs = gridspec.GridSpec(1, 2, width_ratios=[figsize1[0], figsize2[0]])

        # First Subplot
        ax0 = plt.subplot(gs[0])
        ax0.set_title(f'Distribution by {categorical_feature}')
        sns.countplot(data=data, x=categorical_feature, order=data[categorical_feature].value_counts().index, ax=ax0)
        ax0.set_xlabel(categorical_feature)
        ax0.set_ylabel("Count")
        ax0.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax0.patches:
            ax0.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Second(pie chart) Subplot
        ax2 = plt.subplot(gs[1])
        ax2.set_title(f'Distribution by {categorical_feature}')
        data[categorical_feature].value_counts().plot.pie(autopct='%1.1f%%',  shadow=True, ax=ax2)

        # Another Bar chart
        fig2, ax1 = plt.subplots(figsize=figsize3)
        ax1.set_title(f'Distribution by {categorical_feature} and Loan year')
        sns.countplot(data=data, x=categorical_feature, hue='year', order=data[categorical_feature].value_counts().index, ax=ax1)
        ax1.set_xlabel(categorical_feature)
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Show the figures
        st.write(fig)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")


# Function to display frequency distribution table
@st.cache_data
def display_frequency_distribution(dataset, categorical_feature):
    st.write('===================================================================')
    st.write(f'     Percentage distribution of {categorical_feature} feature category based on different CBE_Loan year')
    st.write('===================================================================')

    result = pd.crosstab(
        index=dataset[categorical_feature],
        columns=dataset['year'],
        values=dataset['LOAN_STATUS'],
        aggfunc='count',
        margins=True,  # Added 'margins' for total row/column
        margins_name='Total'  # Custom name for the 'margins' column/row
    )
    result['|'] = '|'

    for year in result.columns[:6]:  # Exclude the last column 'Total'
        result[f'{year}(%)'] = (result[year] / result[year]['Total']) * 100

    # Round the percentage values to 1 decimal place
    result = result.round(1).fillna(0)

    # Display the table using st.dataframe
    st.dataframe(result)
    st.write('====================================================================')

######################## Correlation Analysis ##############################################
# Function to generate the correlation analysis
@st.cache_data
def correlation_analysis(dataset, selected_features, selected_years):
    # Filter dataset based on selected loan years
    selected_data = dataset[dataset['year'].isin(selected_years)]

    # Filter dataset based on selected numerical features
    selected_data = selected_data[selected_features]

    # Calculate the correlation matrix
    corr = selected_data.corr()

    # Calculate p-values
    p_values = pd.DataFrame(index=corr.index, columns=corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            coef, p_value = pearsonr(selected_data.iloc[:, i], selected_data.iloc[:, j])
            p_values.iloc[i, j] = p_value

    # Round correlation matrix to three decimal places
    corr = corr.round(3)

    # Increase the figure size
    plt.figure(figsize=(16, 12))

    # Plot the correlation heatmap
    sns.heatmap(corr, fmt=".3f", cmap='Blues', cbar_kws={'shrink': 0.8})

    # Manually add text annotations for both correlation and p-values
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            text = plt.text(j + 0.5, i + 0.5, f"{corr.iloc[i, j]:.3f}\n(p={p_values.iloc[i, j]:.3f})",
                            ha='center', va='center', color='black', fontsize=10)

    plt.title(f"Correlation Plot of selected Features for Loan Years {', '.join(map(str, selected_years))}")
    st.pyplot(plt)
#########################################################################

# Function to plot line graph for selected feature with Loan_Status
def plot_loan_status_line_graph(data, feature):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")

    try:
        # Convert 'year' to string for consistent plotting
        data['year'] = data['year'].astype(str)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Line plot
        sns.lineplot(data=data, x='year', y=feature, hue='LOAN_STATUS', marker='o', ax=ax)

        ax.set_title(f'Line Graph of {feature} with Loan_Status')
        ax.set_xlabel('Year')
        ax.set_ylabel(feature)
        ax.legend(title='Loan_Status')

        # Show the figure
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting the line graph: {e}")

#########################################################################

# Function to generate a report for a single year with respect to loan status
def generate_single_year_report(data, feature, year, granularity='Yearly', month=None):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")

    try:
        if granularity == 'Yearly':
            # Filter data for the selected year
            year_data = data[data['year'] == year]

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Bar plot
            sns.countplot(data=year_data, x=feature, hue='LOAN_STATUS', ax=ax)
            ax.set_title(f'Distribution of {feature} with Loan_Status for the Year {year}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=80)
            ax.legend(title='Loan_Status')

            # Add numeric annotations on top of the bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

            # Show the figure
            st.pyplot(fig)

        elif granularity == 'Monthly':
            # Ensure 'month' column exists
            if 'month' not in data.columns:
                st.error("The dataset does not contain a 'month' column.")
                return

            # Filter data for the selected year and month
            month_data = data[(data['year'] == year) & (data['month'] == month)]

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Bar plot
            sns.countplot(data=month_data, x=feature, hue='LOAN_STATUS', ax=ax)
            ax.set_title(f'Distribution of {feature} with Loan_Status for {calendar.month_name[month]}/{year}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=80)
            ax.legend(title='Loan_Status')

            # Add numeric annotations on top of the bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

            # Show the figure
            st.pyplot(fig)

        elif granularity == 'Weekly':
            # Ensure 'week' column exists
            if 'week' not in data.columns:
                st.error("The dataset does not contain a 'week' column.")
                return

            # Filter data for the selected year and month
            weekly_data = data[(data['year'] == year) & (data['month'] == month)].groupby(['week', 'LOAN_STATUS'])[feature].count().unstack().fillna(0)

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Stacked bar plot
            weekly_data.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Distribution of {feature} with Loan_Status for {calendar.month_name[month]}/{year} (Weekly)')
            ax.set_xlabel('Week')
            ax.set_ylabel('Count')
            ax.legend(title='Loan_Status')

            # Add numeric annotations on top of the bars
            for i, col in enumerate(weekly_data.columns):
                for idx, val in enumerate(weekly_data[col]):
                    ax.text(idx, val / 2, str(int(val)), color='white', ha='center', va='center')

            # Show the figure
            st.pyplot(fig)

        elif granularity == 'Daily':
            # Ensure 'day' column exists
            if 'day' not in data.columns:
                st.error("The dataset does not contain a 'day' column.")
                return

            # Filter data for the selected year, month and day
            daily_data = data[(data['year'] == year) & (data['month'] == month)].groupby(['day', 'LOAN_STATUS'])[feature].count().unstack().fillna(0)

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Stacked bar plot
            daily_data.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Distribution of {feature} with Loan_Status for {calendar.month_name[month]}/{year} (Daily)')
            ax.set_xlabel('Day')
            ax.set_ylabel('Count')
            ax.legend(title='Loan_Status')

            # Add numeric annotations on top of the bars
            for i, col in enumerate(daily_data.columns):
                for idx, val in enumerate(daily_data[col]):
                    ax.text(idx, val / 2, str(int(val)), color='white', ha='center', va='center')

            # Show the figure
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while generating the report: {e}")

#########################################################################
# Function to plot quarterly reports for a selected feature and year
def plot_quarterly_report(data, feature, year, quarter):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")

    try:
        # Check if 'month' column exists in the dataset
        if 'month' not in data.columns:
            st.error("'month' column is not available in the dataset.")
            return

        # Filter data for the selected year
        data_year = data[data['year'] == year]

        # Convert 'month' to quarter
        data_year['quarter'] = data_year['month'].apply(lambda x: (x - 1) // 3 + 1)

        # Filter data for the selected quarter
        data_quarter = data_year[data_year['quarter'] == quarter]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Count plot for the selected feature by quarter
        sns.countplot(data=data_quarter, x=feature, hue='LOAN_STATUS', ax=ax)

        ax.set_title(f'Distribution of {feature} with Loan_Status for Quarter {quarter} of Year {year}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=80)
        ax.legend(title='Loan_Status')

        # Add numeric annotations on bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # To avoid annotations on empty bars
                ax.annotate(format(height, '.0f'),
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=8, color='black')

        # Show the figure
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting the quarterly report: {e}")

#########################################################################

# Additional styling for the overview section
st.markdown(
    """
    ## Welcome to Loan Status EDA Tool!

    Explore the dynamics of Loan Status in CBE across different Loan years. This interactive tool, powered by exploratory data analysis (EDA), offers insights into key trends.

    ### What You Can Do:
    1. Feature Distribution: Examine feature distributions on both aggregated and by specific loan year.
    2. Bivariate Analysis: Examine feature Selected feature with Loan_status aggregated  year
    3. Correlation Analysis: Understand feature correlations for any CBE year.
    4. Generete report: generate report yearly, monthly, weekly, daily for seelected features with loan_status 
    5. Generate Quarterly Report: Generate reprot Quarterly with specfic year and feature.

    Dive into the rich data of CBE from 2014 to 2024, interact, and uncover valuable insights!
    """
)

# Load cleaned data
cleaned_data_path = 'D:/CBE_project_related/LOAN WITH COLLATERAL/cleaned_data_2014-2024.xlsx'
dataset = load_data(cleaned_data_path)

# List of features to exclude from the dropdown menu
excluded_features = ['year_month']

# Filter out excluded features
allowed_features = [col for col in dataset.columns if col not in excluded_features]

# Function to create a horizontal line with custom styling
def horizontal_line(height=1, color="blue", margin="0.5em 0"): # color="#ddd"
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# Sidebar for selecting parameters
st.sidebar.header('Parameters')

####################### 1. Feature Distribution of UFM ##################################
st.sidebar.markdown('### Feature Distribution')
# Allow the user to select a feature
selected_feature = st.sidebar.selectbox('Select Feature', allowed_features)

# Display distribution plots and tables based on the selected feature
if st.sidebar.button('Show Distribution'):
    st.subheader(f'Frequency Distribution of {selected_feature}')
    display_frequency_distribution(dataset, selected_feature)
    st.subheader(f'Distribution of {selected_feature}')
    plot_categorical_feature(dataset, selected_feature)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

####################### 2. Bivariate Analysis ############################################
st.sidebar.header('Bivariate Analysis')
# Allow the user to select features for bivariate analysis
selected_bivariate_feature = st.sidebar.selectbox('Select Feature for Bivariate Analysis', allowed_features)

# Button to generate line or bar graph for bivariate analysis with Loan_Status
analysis_type = st.sidebar.radio("Select Analysis Type", ("Line Graph", "Bar Graph"))

if st.sidebar.button('Generate Analysis'):
    st.subheader(f'Bivariate Analysis of {selected_bivariate_feature} with LOAN_STATUS')

    if analysis_type == "Line Graph":
        plot_loan_status_line_graph(dataset, selected_bivariate_feature)
    elif analysis_type == "Bar Graph":
        plot_categorical_feature(dataset, selected_bivariate_feature)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

####################### 3. Correlation Analysis ###################################################
st.sidebar.header('Correlation Analysis')
# Filter numerical features only
numerical_features = dataset.select_dtypes(include=['number']).columns

# Allow the user to select the features
selected_features = st.sidebar.multiselect('Select Features for Correlation', numerical_features, key='features')

# Allow the user to select multiple EDHS years using a multiselect
selected_years = st.sidebar.multiselect('Select Loan Years', dataset['year'].unique(), key='years')

# Button to generate correlation matrix
if st.sidebar.button('Generate Correlation Matrix'):
    correlation_analysis(dataset, selected_features, selected_years)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

####################### 4. Single Year Report ###################################################
st.sidebar.header('Single Year Report')
# Allow the user to select a feature and a year
# Allow the user to select a feature and a year
selected_single_year_feature = st.sidebar.selectbox('Select Feature for Single Year Report', allowed_features)
selected_single_year = st.sidebar.selectbox('Select Year for Report', dataset['year'].unique())
granularity = st.sidebar.radio("Select Report Granularity", ("Yearly", "Monthly", "Weekly", "Daily"))

if granularity in ['Monthly', 'Weekly', 'Daily']:
    selected_single_month = st.sidebar.selectbox('Select Month for Report', range(1, 13), format_func=lambda x: calendar.month_name[x])
else:
    selected_single_month = None

# Button to generate the single year report
if st.sidebar.button('Generate Report'):
    st.subheader(f'{granularity} Report of {selected_single_year_feature} for Year {selected_single_year}')
    generate_single_year_report(dataset, selected_single_year_feature, selected_single_year, granularity, selected_single_month)
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
###################5. Quarterly ###############################
# Sidebar for quarterly report
st.sidebar.header('Quarterly Report')
# Allow the user to select a feature for quarterly report
selected_quarterly_feature = st.sidebar.selectbox('Select Feature for Quarterly Report', allowed_features)

# Allow the user to select a year for quarterly report
selected_year = st.sidebar.selectbox('Select Year for Quarterly Report', sorted(dataset['year'].unique()))

# Allow the user to select a quarter for quarterly report
selected_quarter = st.sidebar.selectbox('Select Quarter for Quarterly Report', ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4'])
quarter_mapping = {'Quarter 1': 1, 'Quarter 2': 2, 'Quarter 3': 3, 'Quarter 4': 4}

# Button to generate quarterly report
if st.sidebar.button('Generate Quarterly Report'):
    st.subheader(f'Quarterly Report of {selected_quarterly_feature} for Year {selected_year} ({selected_quarter})')
    if selected_quarter == '':
        plot_quarterly_report(dataset, selected_quarterly_feature, selected_year)
    else:
        plot_quarterly_report(dataset, selected_quarterly_feature, selected_year, quarter_mapping[selected_quarter])

# End of sidebar
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

####################### End of Code ############################
    