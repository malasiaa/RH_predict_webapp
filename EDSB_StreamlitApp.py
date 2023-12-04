import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

filterwarnings("ignore")

def RandomForest(data):
    
    datacols = data.columns

    df = pd.read_csv('HR_DS.csv')
    df = df.drop(['Over18', 'EmployeeNumber','EmployeeCount','StandardHours','MonthlyIncome' ,'YearsInCurrentRole', 'YearsWithCurrManager'],axis=1)

    cat = df.select_dtypes('object')
    for col in cat.columns:
        n = 2
        for i in cat[col].unique():
            df[col] = df[col].replace(i,n)
            n += 1
        
    # normalizing
    scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
    norm = scaler.fit_transform(df)
    norm_df = pd.DataFrame(norm,columns=df.columns)
    
    X = pd.DataFrame(norm_df.drop(columns='Attrition'))
    Y = pd.DataFrame(norm_df.Attrition).values.reshape(-1, 1)
    
    # Drop columns that are not in the specified list
    X = X.drop(columns=X.columns.difference(datacols))

    # Train first and doing oversampling to reduce the imbalance problem
    x_train, x_test, y_train, y_test = train_test_split(X ,Y ,test_size = 0.2 , random_state = 0)
    smote = SMOTE(random_state=0)
    smote_train, smote_target = smote.fit_resample(x_train,y_train)

    # train for Random Forest
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    rfc = RandomForestClassifier()
    rfc = rfc.fit(smote_train , smote_target)
    
    data = data.reindex(columns=X.columns)
    
    y_pred = rfc.predict(data)

    return y_pred

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Attrition Prediction App",
        page_icon=":computer:",
        layout="wide",  # Use "wide" layout for a larger page width
        initial_sidebar_state="expanded",  # Expand the sidebar by default
    )
    
    ################################################################################################################################
    # Custom CSS to position the title and logo side by side
    custom_css = """
        <style>
            .title-logo-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .logo {
                margin-left: 10px;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Title and logo container
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    
    # Logo element
    logo_url = "data:image/png;base64,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"
    logo_html = f'<img src="{logo_url}" alt="Logo" width="100" height="100">'
    st.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)
    
    st.title("Enterprise Data Science Bootcamp Project")
    st.header("Attrition Prediction App")

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)
    ################################################################################################################################

    data = dict()
    warning_msg = "Please enter a valid numeric int value."
    
    # Create two columns
    col1, col2, col3, col4 = st.columns(4)
    
    # First selector box in the first column
    with col1:
        # Dropdown for selecting a value for the variable
        BT_value = st.selectbox('Business Travel', ['','Travel Rarely','Travel Frequently','Non-Travel'])
        if BT_value != '':
            if BT_value == 'Travel Rarely':
                BT_value = 0.0
            elif BT_value == 'Travel Frequently':
                BT_value = 0.5
            elif BT_value == 'Non-Travel':
                BT_value = 1.0
            data.update({'BusinessTravel':BT_value})
            
        Dept_value = st.selectbox('Department', ['','Sales','Research & Development','Human Resources'])
        if Dept_value != '':
            if Dept_value == 'Sales':
                Dept_value = 0.0
            elif Dept_value == 'Research & Development':
                Dept_value = 0.5
            elif Dept_value == 'Human Resources':
                Dept_value = 1.0
            data.update({'Department':Dept_value})
            
        EF_value = st.selectbox('Education Field', ['','Life Sciences','Human Resources','Medical','Marketing','Technical Degree','Other'])
        if EF_value != '':
            if EF_value == 'Life Sciences':
                EF_value = 0.0
            elif EF_value == 'Human Resources':
                EF_value = 1.0
            elif EF_value == 'Medical':
                EF_value = 0.4
            elif EF_value == 'Marketing':
                EF_value = 0.6
            elif EF_value == 'Technical Degree':
                EF_value = 0.8
            elif EF_value == 'Other':
                EF_value = 0.2
            data.update({'EducationField':EF_value})
        
        Gender_value = st.selectbox('Gender', ['','Female','Male'])
        if Gender_value != '':
            if Gender_value == 'Female':
                Gender_value = 0.0
            elif Gender_value == 'Male':
                Gender_value = 1.0
            data.update({'Gender':Gender_value})
        
        JobRole_value = st.selectbox('Job Role', ['','Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director',
                                                'Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources'])
        if JobRole_value != '':
            if JobRole_value == 'Sales Executive':
                JobRole_value = 0.000
            elif JobRole_value == 'Research Scientist':
                JobRole_value = 0.125
            elif JobRole_value == 'Laboratory Technician':
                JobRole_value = 0.250
            elif JobRole_value == 'Manufacturing Director':
                JobRole_value = 0.375
            elif JobRole_value == 'Healthcare Representative':
                JobRole_value = 0.500
            elif JobRole_value == 'Manager':
                JobRole_value = 0.625
            elif JobRole_value == 'Sales Representative':
                JobRole_value = 0.750
            elif JobRole_value == 'Research Director':
                JobRole_value = 0.875
            elif JobRole_value == 'Human Resources':
                JobRole_value = 1.000
            data.update({'JobRole':JobRole_value})
        
        MaritalStatus_value = st.selectbox('Marital Status', ['','Single','Married','Divorced'])
        if MaritalStatus_value != '':
            if MaritalStatus_value == 'Single':
                MaritalStatus_value = 0.0
            elif MaritalStatus_value == 'Married':
                MaritalStatus_value = 0.5
            elif MaritalStatus_value == 'Divorced':
                MaritalStatus_value = 1.0
            data.update({'MaritalStatus':MaritalStatus_value})
        
        OverTime_value = st.selectbox('Over Time', ['','Yes','No'])
        if OverTime_value != '':
            if OverTime_value == 'Yes':
                OverTime_value = 0.0
            elif OverTime_value == 'No':
                OverTime_value = 1.0
            data.update({'OverTime':OverTime_value})
    
    with col2:
        Age_value = st.text_input('Age:', '')
        if not is_numeric(Age_value) and Age_value != '':
            st.warning(warning_msg)
        elif Age_value != '':
            data.update({'Age':Age_value})
        
        DailyRate_value = st.text_input('Daily Rate:', '')
        if not is_numeric(DailyRate_value) and DailyRate_value != '':
            st.warning(warning_msg)
        elif DailyRate_value != '':
            data.update({'DailyRate':DailyRate_value})
        
        DistanceFromHome_value = st.text_input('Distance From Home:', '')
        if not is_numeric(DistanceFromHome_value) and DistanceFromHome_value != '':
            st.warning(warning_msg)
        elif DistanceFromHome_value != '':
            data.update({'DistanceFromHome':DistanceFromHome_value})
        
        Education_value = st.text_input('Education:', '')
        if not is_numeric(Education_value) and Education_value != '':
            st.warning(warning_msg)
        elif Education_value != '':
            data.update({'Education':Education_value})
        
        EnvironmentSatisfaction_value = st.text_input('Environment Satisfaction:', '')
        if not is_numeric(EnvironmentSatisfaction_value) and EnvironmentSatisfaction_value != '':
            st.warning(warning_msg)
        elif EnvironmentSatisfaction_value != '':
            data.update({'EnvironmentSatisfaction':EnvironmentSatisfaction_value})
        
        HourlyRate_value = st.text_input('Hourly Rate:', '')
        if not is_numeric(HourlyRate_value) and HourlyRate_value != '':
            st.warning(warning_msg)
        elif HourlyRate_value != '':
            data.update({'HourlyRate':HourlyRate_value})
        
        
        JobInvolvement_value = st.text_input('Job Involvement:', '')
        if not is_numeric(JobInvolvement_value) and JobInvolvement_value != '':
            st.warning(warning_msg)
        elif JobInvolvement_value != '':
            data.update({'JobInvolvement':JobInvolvement_value})
    
    with col3:
        JobLevel_value = st.text_input('Job Level:', '')
        if not is_numeric(JobLevel_value) and JobLevel_value != '':
            st.warning(warning_msg)
        elif JobLevel_value != '':
            data.update({'JobLevel':JobLevel_value})
        
        JobSatisfaction_value = st.text_input('Job Satisfaction:', '')
        if not is_numeric(JobSatisfaction_value) and JobSatisfaction_value != '':
            st.warning(warning_msg)
        elif JobSatisfaction_value != '':
            data.update({'JobSatisfaction':JobSatisfaction_value})
        
        MonthlyRate_value = st.text_input('Monthly Rate:', '')
        if not is_numeric(MonthlyRate_value) and MonthlyRate_value != '':
            st.warning(warning_msg)
        elif MonthlyRate_value != '':
            data.update({'MonthlyRate':MonthlyRate_value})
        
        NumCompaniesWorked_value = st.text_input('Number Comp. Worked:', '')
        if not is_numeric(NumCompaniesWorked_value) and NumCompaniesWorked_value != '':
            st.warning(warning_msg)
        elif NumCompaniesWorked_value != '':
            data.update({'NumCompaniesWorked':NumCompaniesWorked_value})
        
        PercentSalaryHike_value = st.text_input('Percent Salary Hike:', '')
        if not is_numeric(PercentSalaryHike_value) and PercentSalaryHike_value != '':
            st.warning(warning_msg)
        elif PercentSalaryHike_value != '':
            data.update({'PercentSalaryHike':PercentSalaryHike_value})
        
        PerformanceRating_value = st.text_input('Performance Rating:', '')
        if not is_numeric(PerformanceRating_value) and PerformanceRating_value != '':
            st.warning(warning_msg)
        elif PerformanceRating_value != '':
            data.update({'PerformanceRating':PerformanceRating_value})
        
        RelationshipSatisfaction_value = st.text_input('Relationship Satisfaction:', '')
        if not is_numeric(RelationshipSatisfaction_value) and RelationshipSatisfaction_value != '':
            st.warning(warning_msg)
        elif RelationshipSatisfaction_value != '':
            data.update({'RelationshipSatisfaction':RelationshipSatisfaction_value})
            
    with col4:
        StockOptionLevel_value = st.text_input('Stock Option Level:', '')
        if not is_numeric(StockOptionLevel_value) and StockOptionLevel_value != '':
            st.warning(warning_msg)
        elif StockOptionLevel_value != '':
            data.update({'StockOptionLevel':StockOptionLevel_value})
        
        TotalWorkingYears_value = st.text_input('Total Working Years:', '')
        if not is_numeric(TotalWorkingYears_value) and TotalWorkingYears_value != '':
            st.warning(warning_msg)
        elif TotalWorkingYears_value != '':
            data.update({'TotalWorkingYears':TotalWorkingYears_value})
        
        TrainingTimesLastYear_value = st.text_input('Training Times Last Year:', '')
        if not is_numeric(TrainingTimesLastYear_value) and TrainingTimesLastYear_value != '':
            st.warning(warning_msg)
        elif TrainingTimesLastYear_value != '':
            data.update({'TrainingTimesLastYear':TrainingTimesLastYear_value})
        
        WorkLifeBalance_value = st.text_input('Work Life Balance:', '')
        if not is_numeric(WorkLifeBalance_value) and WorkLifeBalance_value != '':
            st.warning(warning_msg)
        elif WorkLifeBalance_value != '':
            data.update({'WorkLifeBalance':WorkLifeBalance_value})
        
        YearsAtCompany_value = st.text_input('Years At Company:', '')
        if not is_numeric(YearsAtCompany_value) and YearsAtCompany_value != '':
            st.warning(warning_msg)
        elif YearsAtCompany_value != '':
            data.update({'YearsAtCompany':YearsAtCompany_value})
        
        YearsSinceLastPromotion_value = st.text_input('Years Since Last Promotion:', '')
        if not is_numeric(YearsSinceLastPromotion_value) and YearsSinceLastPromotion_value != '':
            st.warning(warning_msg)
        elif YearsSinceLastPromotion_value != '':
            data.update({'YearsSinceLastPromotion':YearsSinceLastPromotion_value})
 
    
    # Button to trigger model prediction
    if st.button("Predict"):
        # Collect input data (replace this with your data collection logic)
        if len(data) == 0:
            prediction_result = 'No prediction. Please input at least one value.'
        else:
            input_data = pd.DataFrame(data,index=[0])
            prediction_result = RandomForest(input_data)
        
        # Display the prediction result
        if int(prediction_result) == 1:
            prediction_result = 'Attrition'
        else:
            prediction_result = 'No Attrition'
            
        st.write(f"Prediction Result: {prediction_result}")
        

def is_numeric(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    main()