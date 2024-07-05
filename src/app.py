import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from langchain_community.llms import GooglePalm
from dotenv import load_dotenv
import os
 # Load environment variables from .env file
#load_dotenv()
import warnings
warnings.filterwarnings("ignore")

# Putting CustomResponseParser class and chat_with_dataset function outside the main_app function
class CustomResponseParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

def chat_with_dataset(df):
    st.subheader(" Chat with Adult Census Income Dataset ⌨️")

    with st.expander("🔎 Dataframe Preview"):
        st.write(df.head(3))

    query = st.text_area("🗣️ Chat with Dataframe", placeholder=" Type your query and press Ctrl + Enter for output: ")
    container = st.container()
    

    if query:
        api_key = 'AIzaSyDlXYnP2xNNa0DNa7dPN89u2L4IuAchEg4'

        # Retrieve the API key from the environment variable
       # api_key = os.getenv('GOOGLEPALM_API_KEY')
       # api_key = os.getenv('GOOGLEPALM_API_KEY')
        #print(f"API Key: '{api_key}' (Length: {len(api_key)})")

        llm = GooglePalm(google_api_key=api_key, temperature=0.2)

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": CustomResponseParser,
            }
        )

        answer = query_engine.chat(query)
        st.write(answer)  # Display the final answer

    # Add some example queries to help users get started
    st.markdown(""" ## Example Queries
    ```python
    -> List the top 5 most common native countries in the dataset.          
    -> What is the average age of individuals based on their marital status?            
    -> Show the top 5 oldest people with Private jobs and hourly weeks equal to 35 and are female.           
    -> Show the distribution of income across different races.            
    -> What is the percentage of people working in state gov jobs?            
    -> Compare the income distribution between all genders            
    -> Do men or women tend to work longer hours per week on average, and how does this correlate with their income levels?              
    -> Show age ,workclass, occupation, and income of 5 males who work for 99 hours per week          
    -> How many individuals are in each relationship category (e.g., Husband, Wife, etc.)?`
                
    NOTE- 
  - You can download the query output as a CSV file by clicking on the download button on the column header.
  - You can search for specific values within the output.
  - You can sort the results by tapping on the respective column headers.           
     """)
    

def main_app(df, df2):
    # Splitting the data
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training and evaluation (using Logistic Regression since it has highest accuracy)
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    # Function to preprocess input data
    def preprocess_input(input_data):
         # Preprocess input data similar to training data preprocessing
        input_data['education'] = input_data['education'].replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],'School')
        input_data['education'] = input_data['education'].replace('HS-grad','High School')
        input_data['education'] = input_data['education'].replace(['Assoc-voc','Assoc-acdm','Prof-school','Some-college'],'Higher-Education')
        input_data['education'] = input_data['education'].replace('Bachelors','Under-Grad')
        input_data['education'] = input_data['education'].replace('Masters','Graduation')
        input_data['education'] = input_data['education'].replace('Doctorate','Doc')
        
        input_data['marital-status'] = input_data['marital-status'].replace(['Married-civ-spouse','Married-AF-spouse'],'Married')
        input_data['marital-status'] = input_data['marital-status'].replace(['Never-married'],'Unmarried')
        input_data['marital-status'] = input_data['marital-status'].replace(['Divorced','Separated','Widowed','Married-spouse-absent'],'Single')
        
        # Dummy encode categorical variables
        categorical_columns = ['education', 'marital-status', 'race', 'gender', 'relationship', 'occupation', 'workclass', 'native-country']
        input_data = pd.get_dummies(input_data, columns=categorical_columns)
        
        # Checking all columns from training data are present
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[X.columns]
        
        return input_data

    # Function to predict income
    def predict_income(input_data):
        processed_data = preprocess_input(input_data)
        scaled_data = scaler.transform(processed_data)
        prediction = logreg_model.predict(scaled_data)[0]
        return prediction

    # Sidebar navigation
    st.sidebar.title(" :orange[Navigate below:] ")
    page = st.sidebar.selectbox("Select a Page:", ["Data Overview", "Visualizations", "Prediction", "Chat with Dataset"])

    st.sidebar.title(":orange[Download Files ⬇️]")
    st.sidebar.link_button("Download Jupyter Notebook", "https://colab.research.google.com/drive/1BZ6W3xpd7zaLoj7dVbLmILvjojpSXACl?usp=sharing")
    st.sidebar.link_button("Download Project Report", "https://drive.google.com/drive/folders/1pNtT3hUaFxUFcrjZcdoJd5e0qcw9DynU?usp=sharing")

    #st.sidebar.markdown("---")
    #st.sidebar.markdown("## Made by Archit Choudhury")
    st.sidebar.title(" :red[By Archit Choudhury] ")
    st.sidebar.markdown(
    """ If you have any questions, feel free to <a href="mailto:architchoudhury10@gmail.com">Send me an email</a>.
    """, unsafe_allow_html=True
        )

    if page == "Data Overview":
        st.write("## Adult Income Dataset Analysis 📊")
        st.write("### Data Overview and Preprocessing")
        st.markdown("`df2.head()`")
        st.write(df2.head())

        st.write("### Dataset Shape")
        st.markdown("`df2.shape`")
        st.write(df2.shape)

        st.write("### Data Types")
        st.markdown("`df2.types`")
        st.write(df2.dtypes)

        st.write("### Missing Values")
        st.markdown("`df2.isnull().sum()`")
        st.write(df2.isnull().sum())

        st.write("### Unique Values in Each Column")
        st.markdown("`df2.nunique()`")
        st.write(df2.nunique())
        
        st.write("### Value Counts for 'workclass'")
        st.markdown("`df2['workclass'].value_counts()`")
        st.write(df2['workclass'].value_counts())

        st.write("### Value Counts for 'occupation'")
        st.markdown("`df2['occupation'].value_counts()`")
        st.write(df2['occupation'].value_counts())

        st.write("### Value Counts for 'native-country'")
        st.markdown("`df2['native-country'].value_counts()`")
        st.write(df2['native-country'].value_counts())

        st.write("### Value Counts for 'income'")
        st.markdown("`df2['income'].value_counts()`")
        st.write(df2['income'].value_counts())

        st.write("### Value Counts for 'marital-status'")
        st.markdown("`df2['marital-status'].value_counts()`")
        st.write(df2['marital-status'].value_counts())

        st.write("### Value Counts for 'gender'")
        st.markdown("`df2['gender'].value_counts()`")
        st.write(df2['gender'].value_counts())

        st.write("### Value Counts for 'race'")
        st.markdown("`df2['race'].value_counts()`")
        st.write(df2['race'].value_counts())

        #st.write("### Cleaning dataset")
        st.markdown("""
             ### Handling Missing Values

              ```python
            # Handling missing values
            df['workclass'] = df['workclass'].replace('?', np.nan)
            df['occupation'] = df['occupation'].replace('?', np.nan)
            df['native-country'] = df['native-country'].replace('?', np.nan)
            df = df.dropna()
            df2 = df.copy()
                        """)
        st.markdown("""
             ### Feature Engineering

        ```python
        # Reducing Columns
        df.education = df.education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'], 'School')
        df.education = df.education.replace('HS-grad', 'High School')
        df.education = df.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'Higher-Education')
        df.education = df.education.replace('Bachelors', 'Under-Grad')
        df.education = df.education.replace('Masters', 'Graduation')
        df.education = df.education.replace('Doctorate', 'Doc')
        
        df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'Married')
        df['marital-status'] = df['marital-status'].replace(['Never-married'], 'Unmarried')
        df['marital-status'] = df['marital-status'].replace(['Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'], 'Single')
        

        df['income'] = df['income'].replace({'<=50K': 0, '>50K': 1})
                     """)

    elif page == "Visualizations":
        # ... (keep the Visualizations code as is)
        st.header("Data Visualizations 👁️👁️")
        st.write("### Correlation Heatmap")
        st.markdown("""
              ```python
                  attributes = ['age', 'hours-per-week', 'educational-num', 'income']
        df_selected = df[attributes]
        corr_matrix = df_selected.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)   
                    """)
        attributes = ['age', 'hours-per-week', 'educational-num', 'income']
        df_selected = df[attributes]
        corr_matrix = df_selected.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

        # st.write("### Income with Education")
        # st.markdown("""
        #       ```python
        # plt.figure(figsize=(10, 6))
        # sns.countplot(x=df2['income'], palette='magma', hue='education', data=df2)
        # st.pyplot(plt)
        #             """)
        
        # plt.figure(figsize=(10, 6))
        # sns.countplot(x=df2['income'], palette='magma', hue='education', data=df2)
        # st.pyplot(plt)

        st.write("### Income Distribution by Workclass")
        plt.figure(figsize=(14, 5))
        sns.histplot(data=df2, x="workclass", hue="income", multiple="stack")
        plt.title("Income Distribution by Workclass")
        plt.xlabel("Workclass")
        plt.ylabel("Count")
        st.pyplot(plt)

        st.write("### Occupation Distribution by Income")
        plt.figure(figsize=(28, 5))
        sns.countplot(data=df2, x="occupation", hue="income", palette='magma')
        plt.title("Occupation Distribution by Income")
        plt.xlabel("Occupation")
        plt.ylabel("Count")
        st.pyplot(plt)

        st.write("### Income Distribution by Marital Status")
        plt.figure(figsize=(18, 5))
        sns.countplot(data=df2, x="marital-status", hue="income")
        plt.title("Income Distribution by Marital Status")
        plt.xlabel("Marital Status")
        plt.ylabel("Count")
        st.pyplot(plt)

        st.write("### Age Distribution by Income")
        plt.figure(figsize=(10, 3))
        sns.histplot(data=df2, x="age", hue="income", kde=True)
        plt.title("Age Distribution by Income")
        plt.xlabel("Age")
        plt.ylabel("Count")
        st.pyplot(plt)

        st.write("### Pie-Chart of Distribution of Workclass Categories")
        st.markdown("""
              ```python workclass_counts = df2['workclass'].value_counts()
        labels = workclass_counts.index
        sizes = workclass_counts.values
        plt.figure(figsize=(8, 14))
        plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140)
        plt.title("Pie-Chart of Distribution of Workclass Categories")
        st.pyplot(plt)
        """)
        workclass_counts = df2['workclass'].value_counts()
        labels = workclass_counts.index
        sizes = workclass_counts.values
        plt.figure(figsize=(8, 14))
        plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140)
        plt.title("Pie-Chart of Distribution of Workclass Categories")
        st.pyplot(plt)

        st.write("### Income Distribution by Gender")
        plt.figure(figsize=(4, 3))
        sns.countplot(data=df2, x="gender", hue="income")
        plt.title("Income Distribution by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.legend()
        st.pyplot(plt)

        st.write("### Income Distribution by Education Level")
        education_order = df2.sort_values('educational-num')['education'].unique()
        plt.figure(figsize=(7, 3))
        sns.countplot(data=df2, x="education", hue="income", order=education_order)
        plt.title("Income Distribution by Education Level")
        plt.xlabel("Education Level")
        plt.ylabel("Count")
        plt.xticks(rotation=60)
        plt.legend(title="Income")
        st.pyplot(plt)

        st.write("### Histogram")
        selected_columns = ['age', 'fnlwgt', 'hours-per-week', 'educational-num', 'income']
        df_selected = df[selected_columns]

        # Calculate number of rows and columns for subplots
        num_cols = len(selected_columns)
        num_rows = math.ceil(num_cols / 3)  # Adjust the divisor (3) based on your preference

        # Plotting histograms for each column
        plt.figure(figsize=(15, 12))
        df_selected.hist(layout=(num_rows, 3), figsize=(15, 12), sharex=False)
        plt.tight_layout()  # Ensures tight layout
        st.pyplot(plt)

        st.write("### Boxplot for Checking Outliers")
        df2.plot(kind='box', figsize=(12, 12), layout=(3, 3), sharex=False, subplots=True)
        st.pyplot(plt)

    elif page == "Prediction":
        st.header('Income Prediction 💰')
        st.write('## Enter Your Details: ')

        # Defining min and max ranges according to dataset
        numeric_ranges = {
            "age": {"min": 17, "max": 90},
            "educational-num": {"min": 1, "max": 16},
            "hours-per-week": {"min": 1, "max": 99}
        }

        # User Input fields with sliders
        age = st.slider('Age', min_value=numeric_ranges["age"]["min"], max_value=numeric_ranges["age"]["max"], value=30)
        education = st.selectbox('Education', ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 
                                            'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 
                                            'Prof-school', 'Doctorate'])
        education_num = st.slider('Years of Education', min_value=numeric_ranges["educational-num"]["min"], max_value=numeric_ranges["educational-num"]["max"], value=10)
        hours_per_week = st.slider('Hours per Week', min_value=numeric_ranges["hours-per-week"]["min"], max_value=numeric_ranges["hours-per-week"]["max"], value=40)
        marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'])
        occupation = st.selectbox('Occupation', ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        relationship = st.selectbox('Relationship', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
        race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        native_country = st.selectbox('Native Country', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands'])

        # Input data preparation
        input_data = pd.DataFrame({
            'age': [age],
            'education': [education],
            'educational-num': [education_num],
            'hours-per-week': [hours_per_week],
            'marital-status': [marital_status],
            'workclass': [workclass],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'native-country': [native_country]
        })

        # Predict income category
        if st.button('Predict Income'):
            prediction = predict_income(input_data)
            
            # Display prediction result
            st.write('## Prediction Result')
            if prediction == 0:
                st.write('### Income is predicted to be <=50K')
            elif prediction == 1:
                st.write('### Income is predicted to be >50K')

    elif page == "Chat with Dataset":
        # Call the chat_with_dataset function
        chat_with_dataset(df2)

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Handling missing values
        df['workclass'] = df['workclass'].replace('?', np.nan)
        df['occupation'] = df['occupation'].replace('?', np.nan)
        df['native-country'] = df['native-country'].replace('?', np.nan)
        df = df.dropna()
        df2 = df.copy()
        
        # Feature engineering
        df.education = df.education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'], 'School')
        df.education = df.education.replace('HS-grad', 'High School')
        df.education = df.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'Higher-Education')
        df.education = df.education.replace('Bachelors', 'Under-Grad')
        df.education = df.education.replace('Masters', 'Graduation')
        df.education = df.education.replace('Doctorate', 'Doc')
        
        df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'Married')
        df['marital-status'] = df['marital-status'].replace(['Never-married'], 'Unmarried')
        df['marital-status'] = df['marital-status'].replace(['Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'], 'Single')
        
        df['income'] = df['income'].replace({'<=50K': 0, '>50K': 1})
        
        df.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['education', 'marital-status', 'race', 'gender', 'relationship', 'occupation', 'workclass', 'native-country'])

        return df, df2
    else:
        st.info("### Waiting for CSV file to be uploaded.")
        return None, None

# Main Streamlit app
st.header(' :blue[INTEL] ML PROJECT BY **:red[ THE SEMICOLONS]** 🌟', divider='rainbow')
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
df, df2 = load_data(uploaded_file)

# Hiding Streamlit hamburger and footer 
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Check if the data is loaded before proceeding
if df is not None and df2 is not None:
    st.write("Data loaded successfully!")
    main_app(df, df2)
else:
    st.warning("### Please upload a CSV file to proceed.")
