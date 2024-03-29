import pandas as pd
import plotly as px

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

def final(df):
    df = df.loc[:, df.columns != 'FLAG']
    
    sex_race_his_obesity = df[df['STUB_NAME'] == 'Sex and race and Hispanic origin']
    race_his_obesity = df[df['STUB_NAME'] == 'Race and Hispanic origin']
    total_obesity = df[df['STUB_NAME'] == 'Total']
    poverty_obesity = df[df['STUB_NAME'] == 'Percent of poverty level']
    sex_obesity = df[df['STUB_NAME'] == 'Sex']
    age_obesity = df[df['STUB_NAME'] == 'Age']

    return sex_race_his_obesity, race_his_obesity, total_obesity, poverty_obesity, sex_obesity, age_obesity
    