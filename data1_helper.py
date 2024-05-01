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
def cleanGenderRace(df):
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
    raceGender_df.loc[:, ["Gender"]] = raceGender_df["STUB_LABEL_NUM"].map(genderDict)
    raceGender_df.loc[:, ["Race"]] = raceGender_df["STUB_LABEL_NUM"].map(raceDict)
    return raceGender_df

def cleanAge(df):
    """
    Clean data 1 to get data containing: age, unit_num = crude, year = 2015-2018
    """

    filtered_df = df[(df["UNIT_NUM"] == 2) & (df["YEAR_NUM"] == 10)]
    age_df = filtered_df[filtered_df["STUB_NAME_NUM"] == 6]
    genderDict = {6.11: 'Male', 6.12: 'Male', 6.13: 'Male', 6.14: 'Male', 6.15: 'Male', 6.16: 'Male',
           6.21: 'Female', 6.22: 'Female', 6.23: 'Female', 6.24: 'Female', 6.25: 'Female', 6.26: 'Female'}
    ageDict = {6.11: '20-34 years', 6.12: '35-44 years', 6.13: '45-54 years', 6.14: '55-64 years', 6.15: '65-74 years', 6.16: '75 years and over',
           6.21: '20-34 years', 6.22: '35-44 years', 6.23: '45-54 years', 6.24: '55-64 years', 6.25: '65-74 years', 6.26: '75 years and over'}
    
    age_df.loc[:, ["Gender"]] = age_df["STUB_LABEL_NUM"].map(genderDict)
    age_df.loc[:, ["Age"]] = age_df["STUB_LABEL_NUM"].map(ageDict)
    return age_df

######## Visualization Helper ########
def obesityRaceVsObesity(raceGender_df):
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

def fitAndEvaluate(model, x, y):
    """
    train the model and evaluate it
    """
    model.fit(x, y)
    print("mean squared error: ", evaluateModel(model, x, y))

def plotML(generalObesity, linRegressML, x):
    """
    plot the best fit model and the data
    """
    ax = sns.scatterplot(generalObesity, x="MiddleYear", y="ESTIMATE")
    ax.set_xlabel(None)
    ax.set_ylabel("Pecentage of Population")
    
    # plot the linear regression best fit line
    predictedY = linRegressML.predict(x)
    predicted2024 = round(linRegressML.predict([2024])[0], 2) # get predicted rate of obesity in 2024
    title = "Consistent Increased rate of obesity in the US.\nThe rate of obesity in the US in 2024 is predicted to be " + str(predicted2024) + "%"
    plt.title(title)
    plt.plot(x, predictedY, 'r')
def poverty_visual(poverty_df):
    poverty_df = df.loc[:,["PANEL", "UNIT", "STUB_NAME", "STUB_LABEL", "YEAR", "AGE", "ESTIMATE", "FLAG", "PANEL_NUM"]]
    poverty_df = poverty_df.loc[poverty_df["UNIT"] != "Percent of population, age-adjusted"]
    poverty_df = poverty_df.loc[poverty_df["STUB_NAME"] == "Percent of poverty level"]
    poverty_df = poverty_df.loc[(poverty_df["PANEL_NUM"] == 1) | (poverty_df["PANEL_NUM"] == 2) | (poverty_df["PANEL_NUM"] == 3)]
    poverty_1988 = poverty_df.loc[poverty_df["YEAR"] == "1988-1994"]
    poverty_2018 = poverty_df.loc[poverty_df["YEAR"] == "2015-2018"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.5)) 
    ax1 = sns.barplot(data=poverty_1988, x="PANEL", y="ESTIMATE", hue="STUB_LABEL", ax=axes[0])
    ax1.set_title('1988')
    ax1.set_xlabel('BMI (Normal to Obese)')
    ax1.set_ylabel('Total Percentage(%)')
    ax1.legend(title='STUB_LABEL')
    ax1.set_xticklabels(['BMI(18.5-24.9)','BMI(25+)', 'BMI(30+)'])
    ax2 = sns.barplot(data=poverty_2018, x="PANEL", y="ESTIMATE", hue="STUB_LABEL", ax=axes[1])
    ax2.set_title('2018')
    ax2.set_xlabel('BMI (Normal to Obese)')
    ax2.set_ylabel('Total Percentage(%)')
    ax2.legend(title='STUB_LABEL')
    ax2.set_xticklabels(['BMI(18.5-24.9)','BMI(25+)', 'BMI(30+)'])
    
    ax1.legend_.set_title('Poverty Levels')
    ax2.legend_.set_title('Poverty Levels')
    fig.suptitle("High-Income Adults in 2018 are more likely to be obese than Low-Income Adults in 1988", fontsize=16, y=1.05)
    plt.show()