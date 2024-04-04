import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def convertPairsListToDict(pairsList):
    returnDict = {}
    for item in pairsList:
        returnDict[item[0]] = item[1]
    return returnDict

def getMapping(df):
    """
       get encoding lists from the data file 
    """
    panelList = df[["PANEL_NUM", "PANEL"]].drop_duplicates().values
    unitList = df[["UNIT_NUM", "UNIT"]].drop_duplicates().values
    stubNameList = df[["STUB_NAME_NUM", "STUB_NAME"]].drop_duplicates().values
    stubLabelList = df[["STUB_LABEL_NUM", "STUB_LABEL"]].drop_duplicates().values
    yearList = df[["YEAR_NUM", "YEAR"]].drop_duplicates().values
    ageList = df[["AGE_NUM", "AGE"]].drop_duplicates().values
    
    panelDict = convertPairsListToDict(panelList)
    unitDict = convertPairsListToDict(unitList)
    stubNameDict = convertPairsListToDict(stubNameList)
    stubLabelDict = convertPairsListToDict(stubLabelList)
    yearDict = convertPairsListToDict(yearList)
    ageDict = convertPairsListToDict(ageList)
    return panelDict, unitDict, stubNameDict, stubLabelDict, yearDict, ageDict

def cleanGenderRace(df):
    encoded_df = df
    filtered_df = encoded_df[(encoded_df["UNIT_NUM"] == 1) & (encoded_df["YEAR_NUM"] == 10)]
    raceGender_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 4]
    genderDict = {3.111: "Male", 3.112: "Female", 3.121: "Male", 3.122: "Female", 3.131: "Male", 3.132: "Female", 3.241: "Male", 3.242: "Female", 3.251: "Male",
           3.252: "Female"}
    raceDict = {3.111: "White", 3.112: "White", 3.121: "Black or African American", 3.122: "Black or African American", 3.131: "Asian", 3.132: "Asian", 
                3.241: "Hispanic or Latino", 3.242: "Hispanic or Latino", 3.251: "Hispanic or Latino: Mexican origin", 3.252: "Hispanic or Latino: Mexican origin"}
    raceGender_df["Gender"] = raceGender_df["STUB_LABEL_NUM"].map(genderDict)
    raceGender_df["Race"] = raceGender_df["STUB_LABEL_NUM"].map(raceDict)

    return raceGender_df

def cleanAge(df):
    encoded_df = df
    filtered_df = encoded_df[(encoded_df["UNIT_NUM"] == 2) & (encoded_df["YEAR_NUM"] == 10)]
    age_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 6]
    genderDict = {6.11: 'Male', 6.12: 'Male', 6.13: 'Male', 6.14: 'Male', 6.15: 'Male', 6.16: 'Male',
           6.21: 'Female', 6.22: 'Female', 6.23: 'Female', 6.24: 'Female', 6.25: 'Female', 6.26: 'Female'}
    ageDict = {6.11: '20-34 years', 6.12: '35-44 years', 6.13: '45-54 years', 6.14: '55-64 years', 6.15: '65-74 years', 6.16: '75 years and over',
           6.21: '20-34 years', 6.22: '35-44 years', 6.23: '45-54 years', 6.24: '55-64 years', 6.25: '65-74 years', 6.26: '75 years and over'}
    
    age_df["Gender"] = age_df["STUB_LABEL_NUM"].map(genderDict)
    age_df["Age"] = age_df["STUB_LABEL_NUM"].map(ageDict)
    return age_df

def cleanPoverty(df):
    encoded_df = df
    filtered_df = encoded_df[(encoded_df["UNIT_NUM"] == 1) & (encoded_df["YEAR_NUM"] == 10)]
    poverty_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 5]
    return poverty_df

def obesityRaceVObesity(raceGender_df):
    # race vs each level of obesity:
    obesity1_df = raceGender_df[raceGender_df["PANEL_NUM"] == 4]
    raceObesity1 = obesity1_df[["Gender", "Race", "ESTIMATE", "SE"]]
    
    obesity2_df = raceGender_df[raceGender_df["PANEL_NUM"] == 5]
    raceObesity2 = obesity2_df[["Gender", "Race", "ESTIMATE", "SE"]]
    
    obesity3_df = raceGender_df[raceGender_df["PANEL_NUM"] == 6]
    raceObesity3 = obesity3_df[["Gender", "Race", "ESTIMATE", "SE"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax1 = sns.barplot(ax=axes[0], data=raceObesity1, x="Race", y="ESTIMATE", hue="Gender", palette=["#2986cc", "#c90076"])
    ax2 = sns.barplot(ax=axes[1], data=raceObesity2, x="Race", y="ESTIMATE", hue="Gender", palette=["#2986cc", "#c90076"])
    ax3 = sns.barplot(ax=axes[2], data=raceObesity3, x="Race", y="ESTIMATE", hue="Gender", palette=["#2986cc", "#c90076"])
    ax1.set_title("Grade 1 Obesity (30.0 <= BMI <= 34.9)")
    ax1.set_ylabel("Percentage of Population")
    ax1.set_xlabel(None)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=35, ha="right")
    ax2.set_title("Grade 2 Obesity (35.0 <= BMI <= 39.9)")
    ax2.set_ylabel("Percentage of Population")
    ax2.set_xlabel(None)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35, ha="right")
    ax3.set_title("Grade 3 Obesity (BMI >=40.0)")
    ax3.set_ylabel("Percentage of Population")
    ax3.set_xlabel(None)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=35, ha="right")
    
    fig.suptitle("Asians have less obesity rate than people of other races in all levels of Obesity", fontsize=16, y=1.05)













