import pandas as pd

def clean_data(data_path):
    df = pd.read_csv(data_path)

    df = df.drop(['Data_Value_Unit', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'DataValueTypeID', 'QuestionID', 'TopicID', 'ClassID'], axis=1)
    df = df.dropna(subset=['GeoLocation'])
    df = df.dropna(subset=['Data_Value'])
    df = df.dropna(subset=['Age(years)', 'Gender', 'Race/Ethnicity'], how='all')

    for column in ['Age(years)', 'Gender', 'Race/Ethnicity', 'Income']:
        df[column] = df[column].fillna('Missing Data')

    year_comparison = (df['YearStart'] == df['YearEnd']).all()
    if year_comparison:
        df = df.drop('YearEnd', axis=1)
    return df

if __name__ == "__main__":
    data_path = "Datasets/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv"
    clean_data(data_path)
