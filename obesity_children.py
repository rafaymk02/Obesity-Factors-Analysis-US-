import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def remove_single_unique(df):
    uniques = []
    
    col_names = list(df.columns)
    for col in col_names:
        if (len(df[col].unique())) != 1:
            uniques.append(col)

    return df.loc[:, uniques]
    
def remove_num_columns(df):
    nonNum = []
    
    col_names = list(df.columns)
    for col in col_names:
        if "_NUM" not in col :
            nonNum.append(col)
    return df.loc[:, nonNum]

def final_clean(df):
    sex_race_his_obesity = df[df['STUB_NAME'] == 'Sex and race and Hispanic origin']
    race_his_obesity = df[df['STUB_NAME'] == 'Race and Hispanic origin']
    total_obesity = df[df['STUB_NAME'] == 'Total']
    poverty_obesity = df[df['STUB_NAME'] == 'Percent of poverty level']
    sex_obesity = df[df['STUB_NAME'] == 'Sex']
    age_obesity = df[df['STUB_NAME'] == 'Age']
    return sex_race_his_obesity, race_his_obesity, total_obesity, poverty_obesity, sex_obesity, age_obesity

def obesity_EDA_visual(df):
    df['END_YEAR'] = pd.to_datetime(df['YEAR'].str.split('-', expand=True)[0], format='%Y')
    plt.plot(df['END_YEAR'], df['ESTIMATE'], marker='o', label='Obesity Rate')
    plt.plot(df['END_YEAR'], df['SE'], marker='.', label='Standard Error')
    plt.title('Obesity Rates for Individuals aged 2-19')
    plt.xlabel('Year')
    plt.ylabel('Obesity Rate Estimate(%)')
    plt.legend() 
    plt.grid(True)
    plt.show()


def obesity_Visual1(df):
    df = df.loc[df['AGE'] != '2-19 years']
    df['AGE'] = pd.Categorical(df['AGE'], categories=['2-5 years', '6-11 years', '12-19 years'], ordered = False)
    df['STUB_LABEL'] = pd.Categorical(df['STUB_LABEL'], categories=['Below 100%', '100%-199%', '200%-399%', '400% or more'], ordered = False)
    estimatePerYear = (df.groupby(['STUB_LABEL', 'AGE'])['ESTIMATE'].sum() / df.groupby(['STUB_LABEL', 'AGE'])['ESTIMATE'].count()).reset_index()
    fig = px.bar(estimatePerYear, x="STUB_LABEL", y="ESTIMATE", color="AGE",
                 title="Adolescents Below the Poverty Line Face Highest Obesity Rates from Ages 2-19 ",
                 labels={'STUB_LABEL': 'Poverty Levels (Highest to Least)', 'ESTIMATE': 'Obesity Rate (%)'},
                 barmode='group')

    fig.show()

def obesity_Visual2(df):
    df = df.loc[df['AGE'] == '2-19 years']
    df.rename(columns = {"STUB_LABEL": "RACE"}, inplace = True)
    fig = px.line(df, x="YEAR", y="ESTIMATE", color="RACE", title="Obesity Percentage Per Year Based On Race")
    
    fig.update_xaxes(title_text="YEARS", tickfont=dict(size=10))
    fig.update_yaxes(title_text="Obesity rate (%)")
    
    fig.update_traces(line=dict(width=2.75))
    fig.show()




    
    