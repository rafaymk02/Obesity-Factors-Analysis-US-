import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(df):
    # Select relevant features and target variable
    features = ['Age(years)', 'Gender', 'Race/Ethnicity']
    target = 'Data_Value'

    # Filter the dataset to include only physical activity-related data
    activity_df = df[(df['Topic'] == 'Physical Activity - Behavior') & (df['Question'].str.contains('no leisure-time physical activity'))]

    # Convert 'Data_Value' to a binary variable based on a threshold
    threshold = 30
    activity_df = activity_df.copy()
    activity_df['Inactive'] = (activity_df['Data_Value'] >= threshold).astype(int)

    # Use 'Inactive' as the target
    target = 'Inactive'

    # Perform one-hot encoding for categorical features
    activity_df_encoded = pd.get_dummies(activity_df[features], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(activity_df_encoded, activity_df[target], test_size=0.2, random_state=42)

    # Create and train the Logistic Regression model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    return lr_model, X_test, y_test

def evaluate_model(lr_model, X_test, y_test):
    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_mat)

def predict_inactivity(lr_model, sample_inputs):
    # Get the feature names from the trained model
    feature_names = lr_model.feature_names_in_

    # Predict inactivity for each sample input
    for i, sample_input in enumerate(sample_inputs, start=1):
        # Create a DataFrame with all the encoded features from the trained model
        sample_df = pd.DataFrame(columns=feature_names, index=[0])
        sample_df.fillna(0, inplace=True)

        # Set the corresponding values based on the sample input
        for feature, value in sample_input.items():
            encoded_feature = f"{feature}_{value}"
            if encoded_feature in feature_names:
                sample_df[encoded_feature] = 1

        # Make a prediction for the sample input
        sample_prediction = lr_model.predict(sample_df)

        print(f"\nSample Input {i}:")
        print(f"Age: {sample_input['Age(years)']}")
        print(f"Gender: {sample_input['Gender']}")
        print(f"Race/Ethnicity: {sample_input['Race/Ethnicity']}")
        print(f"Predicted Inactivity: {'Inactive' if sample_prediction[0] == 1 else 'Active'}")

def baseline_model(y_test, y_pred):
    # Calculate the median value of the target variable
    median_inactive = y_test.median()

    # Create baseline predictions based on the median value
    baseline_predictions = [1 if x >= median_inactive else 0 for x in y_test]

    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # Calculate baseline confusion matrix
    baseline_confusion_matrix = confusion_matrix(y_test, baseline_predictions)

    print("Baseline Accuracy:", baseline_accuracy)
    print("Baseline Confusion Matrix:")
    print(baseline_confusion_matrix)

    # Evaluate the logistic regression model
    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_confusion_matrix = confusion_matrix(y_test, y_pred)

    print("\nLogistic Regression Accuracy:", lr_accuracy)
    print("Logistic Regression Confusion Matrix:")
    print(lr_confusion_matrix)