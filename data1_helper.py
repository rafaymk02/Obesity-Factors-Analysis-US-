import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.metrics as sm

######## Data understanding Helper ########
def getInfo(df):
    """
    Get general information about the data that will be helpful for cleaning data
    """
    
    print("Info about the data:")
    df.info()
    print("\nNum Unique values per column:")
    print(df.nunique())

def convertPairsListToDict(pairsList):
    """
    Convert list of pair (key, value) to dictionary {key: value) and return that dictionary
    """
    
    returnDict = {}
    for item in pairsList:
        returnDict[item[0]] = item[1]
    return returnDict

def getMapping(df):
    """
    Get encoding dictionarys from the data
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

######## Cleaning Data Helper ########
def cleanData1(df):
    """
    Clean data 1 to get data containing: gender, race, year = 2015-2018 , unit = age-adjusted
    """
    
    filtered_df = df[(df["UNIT_NUM"] == 1) & (df["YEAR_NUM"] == 10)]
    raceGender_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 4]
    genderDict = {3.111: "Male", 3.112: "Female", 3.121: "Male", 3.122: "Female", 3.131: "Male", 3.132: "Female", 3.241: "Male", 3.242: "Female", 
                  3.251: "Male", 3.252: "Female"}
    raceDict = {3.111: "White", 3.112: "White", 3.121: "Black or African American", 3.122: "Black or African American", 3.131: "Asian", 
                3.132: "Asian", 3.241: "Hispanic or Latino", 3.242: "Hispanic or Latino", 3.251: "Hispanic or Latino: Mexican origin", 
                3.252: "Hispanic or Latino: Mexican origin"}
    raceGender_df["Gender"] = raceGender_df["STUB_LABEL_NUM"].map(genderDict)
    raceGender_df["Race"] = raceGender_df["STUB_LABEL_NUM"].map(raceDict)
    return raceGender_df

def cleanData2(df):
    """
    Clean data 1 to get data containing: age, unit_num = crude, year = 2015-2018
    """

    filtered_df = df[(df["UNIT_NUM"] == 2) & (df["YEAR_NUM"] == 10)]
    age_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 6]
    genderDict = {6.11: 'Male', 6.12: 'Male', 6.13: 'Male', 6.14: 'Male', 6.15: 'Male', 6.16: 'Male',
           6.21: 'Female', 6.22: 'Female', 6.23: 'Female', 6.24: 'Female', 6.25: 'Female', 6.26: 'Female'}
    ageDict = {6.11: '20-34 years', 6.12: '35-44 years', 6.13: '45-54 years', 6.14: '55-64 years', 6.15: '65-74 years', 6.16: '75 years and over',
           6.21: '20-34 years', 6.22: '35-44 years', 6.23: '45-54 years', 6.24: '55-64 years', 6.25: '65-74 years', 6.26: '75 years and over'}
    
    age_df["Gender"] = age_df["STUB_LABEL_NUM"].map(genderDict)
    age_df["Age"] = age_df["STUB_LABEL_NUM"].map(ageDict)
    return age_df

######## Machine learning Helper ########
def getMiddleYear(yearDict):
    middleYearDict = {}
    for key, value in yearDict.items():
        dashInd = value.find("-")
        firstYear = int(value[:dashInd])
        secondYear = int(value[dashInd+1:])
        middleYear = int((firstYear + secondYear) / 2)
        middleYearDict[key] = middleYear
    return middleYearDict

def getFilteredData(df, unitNum, panelNum, stubNameNum, stubLabelNum, middleYearDict):
    """
    Filtered the df based on input UNIT_NUM, PANEL_NUM, STUB_NAME_NUM, STUB_LABEL_NUM
    """
    df = df[(df["UNIT_NUM"]) & (df["PANEL_NUM"] == panelNum) & (df["STUB_NAME_NUM"] == stubNameNum) & (df["STUB_LABEL_NUM"] == stubLabelNum)]
    df = df[~pd.isna(df["ESTIMATE"])]
    df = df[(df["FLAG"] == '.') | (pd.isna(df["FLAG"]))]
    df["MiddleYear"] = df["YEAR_NUM"].map(middleYearDict)
    return df

def evaluateModel(model, x, y):
    """
    Use mean squared error to evalue the input model
    Return mean squared error
    """
    predictedY = model.predict(x)
    return sm.mean_squared_error(y, predictedY)

class MeanModel():
    """
    A baseline machine learning model that predicts result based on the mean of the training data
    """
    def __init__(self):
        predictedValue = None

    def fit(self, x, y):
        """
        Implement fit by taking the average of y
        """
        self.predictedValue = y.mean()

    def predict(self, x):
        """
        Predict rate of obesity based on the input year x
        """
        predictedY = []
        for i in x:
            predictedY.append(self.predictedValue)
        return predictedY

class LinearRegressionModel():
    """
    Linear regression model that predicts result based on the best fit line for training data
    """
    def __init__(self):
        slope = None
        intercept = None

    def fit(self, x, y):
        """
        Train the model using linear regression algorithm
        """
        slope, intercept, _, _, _ = stats.linregress(x, y)
        self.slope = slope
        self.intercept = intercept

    def predict(self, x):
        """
        Predict rate of obesity based on the input year x
        """
        return list(map(lambda x: self.slope * x + self.intercept, x))
