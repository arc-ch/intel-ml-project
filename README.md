# 🚀Adult Income Machine Learning Project for Intel 

Welcome to the Adult Income Machine Learning Project. This project aims to analyze the Adult Census Income dataset and build a predictive model for income classification using Logistic Regression. The app also includes features for data visualization and an interactive chat interface for querying the dataset. 

  ### 🔗LIVE NOW- 
  #### Version 1: (Chat with Dataset shows PaLM2 Deprecation errors)
  https://archit-adult-income.streamlit.app/ OR https://archit-adult-income.onrender.com/   (SLOW)
  #### Version 2: (Chat with Dataset working)
  https://archit-adult-income-v2.streamlit.app/ ( Gemini API entered by User)


## Dataset 📊
The dataset used in this project is the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult). It is a public dataset provided by the UCI Machine Learning Repository, containing demographic information about individuals along with their income levels.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
   - [Data Overview](#data-overview)
   - [Visualizations](#visualizations)
   - [Prediction](#prediction)
   - [Chat with Dataset](#chat-with-dataset)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the App](#running-the-app)
6. [Important Notice: API and LangChain Considerations](#important-notice-api-and-langchain-considerations)


##  1. Project Overview 📝
The Project uses the Adult Census Income dataset to explore various demographic features and their relationships with income levels. The app provides a comprehensive set of tools for data analysis, visualization, and prediction.

##  2. Features 🛠️
  ### 1. Data Overview
  Insert adult.csv dataset to proceed 
  
  ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/34e57e25-2133-49a6-98a3-08cda7e4f4c5)

  Explore the dataset's structure, including summary statistics, missing values, and unique value counts for each feature.
  
  ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/7db2a056-c3a8-43ec-bd8a-d953c2902d2b)
  
  
  ### 2. Visualizations 📊
  Generate insightful visualizations such as correlation heatmaps, distribution plots, and more:
  - Correlation Heatmap
    
    ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/d43eb6a4-314e-4d85-9126-3bbdbabc61a2)
  - Income Distribution by Workclass, Occupation, Marital Status, Gender, Race and Education Level
    
    ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/a8d99e75-94d4-4114-bf92-6e421edde3c9)
  
  - Age Distribution by Income
  - Pie-Chart of Workclass Distribution
  - Histograms and Boxplots for numerical features
  
  ### 3. Prediction 🎯
  Predict income based on user-provided demographic information such as age, education, work hours, marital status, workclass, occupation, relationship, race, gender, and native country.
  
  ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/de2b011e-476b-47b7-b640-975bb00e9ecd)
  ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/cc8ac15a-5409-4d0d-a05d-37c775280a45)
  
  
  
  ### 4. Chat with Dataset 💬
  Interactively query the dataset using natural language to get answers and visualizations based on your questions.
  
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/9059dd09-a437-4422-8949-4b6c06cd5b55)
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/fe8ce178-4e53-4adb-911c-3f1498aaffa9)
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/ff5c7f79-301d-497e-886e-7da57535eb35)
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/2beb7df1-dea6-486c-ac9b-b5597b3d91a1)
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/d259a36f-d718-4c70-b0fc-657cfd51dbe5)
 ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/a8d7da49-68e5-4f35-a248-941f0afa8c90)

 


 
## 3. Requirements
Ensure you have the following Python packages installed:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandasai
- langchain_community
- python-dotenv
- google-generativeai

## 4. Installation 🛠️
Clone the repository:
```sh
git clone https://github.com/yourusername/intel-ml-project.git
cd intel-ml-project
```
Install the required packages:

```sh
pip install -r requirements.txt
```

## 5. Running the App
To run the Streamlit app, execute the following command:
```sh
streamlit run app.py
```

  ### Example Queries
  Here are some example queries you can try in the "Chat with Dataset" section:
  
  -  List the top 5 most common native countries in the dataset.
  -  What is the average age of individuals based on their marital status?
  -  Show the top 5 oldest people with Private jobs and hourly weeks equal to 35 and are female.
  -  Show the distribution of income across different races.
  -  What is the percentage of people working in state gov jobs?
  -  Compare the income distribution between all genders.
  -  Do men or women tend to work longer hours per week on average, and how does this correlate with their income levels?
  -  Show age, workclass, occupation, and income of 5 males who work for 99 hours per week.
  -  How many individuals are in each relationship category (e.g., Husband, Wife, etc.)?  
  
 ### Note:
  - You can download the query output as a CSV file by clicking on the download button on the column header.
  - You can search for specific values within the output.
  - You can sort the results by tapping on the respective column headers.


## 6. Important Notice: API and LangChain Considerations

  ### API Reliability 🛠️
  
  The "Chat with Dataset" feature relies on the GenAI API (formerly GooglePalm), which may experience downtime or disruptions. If the API is not functioning, you can view a demo [here]
  
  ### LangChain Deprecation ⚠️
  
  This application uses LangChain for integration with language models like GenAI. Please be aware that LangChain could undergo changes or be deprecated in the future. Stay updated with the [LangChain documentation](https://langchain.readthedocs.io/en/latest/) for compatibility and support information.
  ![image](https://github.com/arc-ch/intel-ml-project/assets/134518231/a5b9ae74-eba6-43b1-94db-15adb64f7306)

  For long-term reliability:
  - Update dependencies regularly and test with the latest versions.
  - Monitor GenAI API and LangChain repositories for updates and announcements.
  - Consider implementing fallback mechanisms for continuity in case of disruptions.
