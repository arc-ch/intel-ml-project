# INTEL Machine Learning Project 🚀

Welcome to the INTEL Machine Learning Project by The Semicolons! This project aims to analyze the Adult Census Income dataset and build a predictive model for income classification using Logistic Regression. The app also includes features for data visualization and an interactive chat interface for querying the dataset.

## Dataset 📊
The dataset used in this project is the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult). It is a public dataset provided by the UCI Machine Learning Repository, containing demographic information about individuals along with their income levels.

## Table of Contents
1. Project Overview
2. Features
3. Usage
4. Data Overview
5. Visualizations
6. Prediction
7. Chat with Dataset
8. Requirements
9. Installation
10. Running the App

## Project Overview 📝
The INTEL project uses the Adult Census Income dataset to explore various demographic features and their relationships with income levels. The app provides a comprehensive set of tools for data analysis, visualization, and prediction.

## Features 🛠️
### Data Overview
Explore the dataset's structure, including summary statistics, missing values, and unique value counts for each feature.

### Visualizations 📊
Generate insightful visualizations such as correlation heatmaps, distribution plots, and more:
- Correlation Heatmap
- Income Distribution by Workclass, Occupation, Marital Status, Gender, and Education Level
- Age Distribution by Income
- Pie-Chart of Workclass Distribution
- Histograms and Boxplots for numerical features

### Prediction 🎯
Predict income based on user-provided demographic information such as age, education, work hours, marital status, workclass, occupation, relationship, race, gender, and native country.

### Chat with Dataset 💬
Interactively query the dataset using natural language to get answers and visualizations based on your questions.

## Usage ℹ️
### Requirements
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

### Installation 🛠️
Clone the repository:
```sh
git clone https://github.com/yourusername/intel-ml-project.git
cd intel-ml-project
```
Install the required packages:

```sh
pip install -r requirements.txt
```

## Running the App
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
Note:
- You can download the query output as a CSV file by clicking on the download button on the column header.
- You can search for specific values within the output.
- You can sort the results by tapping on the respective column headers.
