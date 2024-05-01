import pandas as pd
import plotly.express as px
from ipywidgets import interact, Dropdown

#functions to load data
def load_and_prepare_data():
    df = pd.read_csv('Datasets/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
    obesity_df = df[(df['Class'] == 'Obesity / Weight Status') &
                    (df['Question'] == 'Percent of adults aged 18 years and older who have obesity')]
    obesity_df = obesity_df.dropna(subset=['Data_Value', 'Race/Ethnicity'])
    total_obesity = obesity_df.groupby(['LocationAbbr', 'LocationDesc', 'YearStart']).agg({'Data_Value':'median'}).reset_index()
    total_obesity['Race/Ethnicity'] = 'Total'
    return pd.concat([obesity_df, total_obesity])

def update_plot(race, concatenated_df):
    filtered_data = concatenated_df[concatenated_df['Race/Ethnicity'] == race]
    fig = px.choropleth(filtered_data,
                        locations="LocationAbbr",
                        locationmode="USA-states",
                        color="Data_Value",
                        hover_name="LocationDesc",
                        color_continuous_scale=['yellow', 'red'],
                        labels={'Data_Value': 'Obesity Rate'},
                        title=f"Obesity Rates by {race}",
                        scope="usa")
    fig.show()

if __name__ == "__main__":
    concatenated_df = load_and_prepare_data()
    races_with_total = concatenated_df['Race/Ethnicity'].dropna().unique()
    race_dropdown = Dropdown(options=races_with_total, description="Select Race:")
    interact(update_plot, race=race_dropdown, concatenated_df=fixed(concatenated_df))
