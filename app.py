import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
import category_encoders as ce
import random

# Read data from the data directory
df = pd.read_csv('Fashion_Recommendation_1.csv')

# Rename columns and remove unnecessary columns
df.rename(columns={'Timestamp': 'Timestamp', 
                   'Score': 'Score', 
                   'Gender': 'Gender', 
                   'Age': 'Age', 
                   'Skin Color': 'Skin Color',
                   'What kind of clothes do you feel comfortable wearing?': 'Comfortable Clothes', 
                   'Your Name': 'Name', 
                   'What color do you like to wear in Punjabi?': 'Punjabi Color',
                   'What color do you like to wear in Shirt?': 'Shirt Color', 
                   'What color do you like to wear in Saree?': 'Saree Color',
                   'What color do you like to wear in Kamiz?': 'Kamiz Color', 
                   'Unnamed: 11': 'Unused',
                   'What color clothes do you like to wear?': 'Other Preferred Color', 
                   'Email Address': 'Email Address'}, inplace=True)

df.drop(columns=['Unused','Timestamp','Email Address','Score','Name','Other Preferred Color'], inplace=True)

# Remove Bangla characters
df = df.apply(lambda x: x.str.split('(', expand=True)[0])

# Filter the DataFrame to remove rows where 'Comfortable Clothes' count is less than 5
df = df[df.groupby('Comfortable Clothes')['Comfortable Clothes'].transform('count') >= 5]

# Handle missing value 'Gender' column
df['Gender'] = df['Gender'].str.strip()
df['Gender'].fillna('Female', inplace=True)

# Replace values in the 'Age' column
df['Age'].replace({'12-22': '12-23'}, inplace=True)

# Define gender-specific color preferences
male_colors = {'Panjabi': ['Gray', 'Blue', 'White'], 'Shirt': ['Black', 'Blue', 'White']}
female_colors = {'Saree': ['White', 'Blue', 'Black'], 'Kamiz': ['Green', 'Red', 'Black']}

# Define function to fill null values with random colors
def fill_null_color(row):
    colors = []
    if pd.isnull(row['Punjabi Color']):
        if row['Gender'] == 'Male' and row['Comfortable Clothes'] == 'Panjabi':
            colors.append(random.choice(['Gray', 'Blue', 'White']))
        else:
            colors.append('')
    else:
        colors.append(row['Punjabi Color'])
        
    if pd.isnull(row['Shirt Color']):
        if row['Gender'] == 'Male' and row['Comfortable Clothes'] == 'Shirt':
            colors.append(random.choice(['Black', 'Blue', 'White']))
        else:
            colors.append('')
    else:
        colors.append(row['Shirt Color']) 
        
    if pd.isnull(row['Saree Color']):
        if row['Gender'] == 'Female' and row['Comfortable Clothes'] == 'Saree':
            colors.append(random.choice(['White', 'Blue', 'Black']))
        else:
            colors.append('')
    else:
        colors.append(row['Saree Color'])
        
    if pd.isnull(row['Kamiz Color']):
        if row['Gender'] == 'Female' and row['Comfortable Clothes'] == 'Salwar Kamiz':
            colors.append(random.choice(['Green', 'Red', 'Black']))
        else:
            colors.append('')
    else:
        colors.append(row['Kamiz Color'])
    
    return pd.Series(colors, index=['Punjabi Color', 'Shirt Color', 'Saree Color', 'Kamiz Color'])

# Apply the function to fill null values in color columns
df[['Punjabi Color', 'Shirt Color', 'Saree Color', 'Kamiz Color']] = df.apply(fill_null_color, axis=1)

# Define features and target variables
X = df[['Gender', 'Age', 'Skin Color']]
y_clothes = df['Comfortable Clothes']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable
y_clothes_encoded = label_encoder.fit_transform(y_clothes)

# Split the dataset into training and testing sets
X_train, X_test, y_clothes_train, y_clothes_test = train_test_split(X, y_clothes_encoded, test_size=0.2, random_state=42)

# Initialize TargetEncoder
encoder = ce.TargetEncoder(cols=['Gender', 'Age', 'Skin Color'])

# Fit and transform the training data
X_train_encoded = encoder.fit_transform(X_train, y_clothes_train)

# Transform the testing data
X_test_encoded = encoder.transform(X_test)

# Initialize RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_encoded, y_clothes_train)

# Make predictions on the testing data
y_clothes_pred = clf.predict(X_test_encoded)

# Define a Streamlit app
def main():
    st.title("Comfortable Clothes and Color Prediction App")

    # Display user inputs
    st.sidebar.title("User Inputs")
    gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
    age = st.sidebar.selectbox("Age", df['Age'].unique())
    skin_color = st.sidebar.selectbox("Skin Color", df['Skin Color'].unique())

    # Make prediction based on user inputs
    input_data = {
        'Gender': gender,
        'Age': age,
        'Skin Color': skin_color
    }

    input_df = pd.DataFrame(input_data, index=[0])
    input_encoded = encoder.transform(input_df)

    prediction = label_encoder.inverse_transform(clf.predict(input_encoded))[0]

    if st.button("Predict"):
        st.write(f"Predicted Comfortable Clothes: {prediction}")
        predict_color(prediction, input_data)

def predict_color(clothes, user_input):
    st.write("Predicting Color...")
    if clothes == 'Panjabi':
        predicted_color = Punjabi_Color(user_input, df)
    elif clothes == 'Shirt':
        predicted_color = Shirt_Color(user_input, df)
    elif clothes == 'Saree':
        predicted_color = Saree_Color(user_input, df)
    elif clothes == 'Salwar Kamiz':
        predicted_color = Kamiz_Color(user_input, df)
    else:
        predicted_color = "No color prediction function available for the predicted comfortable clothes."
    st.write(f"Predicted Color: {predicted_color}")

def Punjabi_Color(user_input, df):
    # Define features and target variables
    Punjabi_df = df[df['Comfortable Clothes'] == 'Panjabi']
    X = Punjabi_df[['Gender', 'Age', 'Skin Color']]
    y_color = Punjabi_df['Punjabi Color']

    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode the target variable
    y_color_encoded = label_encoders['Gender'].transform(y_color)

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X, y_color_encoded)

    # Transform user input
    for column in user_input:
        user_input[column] = label_encoders[column].transform([user_input[column]])[0]

    # Make prediction
    predicted_color_encoded = clf.predict(pd.DataFrame(user_input, index=[0]))
    predicted_color = label_encoders['Gender'].inverse_transform(predicted_color_encoded)[0]
    
    return predicted_color

def Shirt_Color(user_input, df):
    # Define features and target variables
    shirt_df = df[df['Comfortable Clothes'] == 'Shirt']
    X = shirt_df[['Gender', 'Age', 'Skin Color']]
    y_color = shirt_df['Shirt Color']

    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode the target variable
    y_color_encoded = label_encoders['Gender'].transform(y_color)

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X, y_color_encoded)

    # Transform user input
    for column in user_input:
        user_input[column] = label_encoders[column].transform([user_input[column]])[0]

    # Make prediction
    predicted_color_encoded = clf.predict(pd.DataFrame(user_input, index=[0]))
    predicted_color = label_encoders['Gender'].inverse_transform(predicted_color_encoded)[0]
    
    return predicted_color

def Saree_Color(user_input, df):
    saree_df = df[df['Comfortable Clothes'] == 'Saree']
    X = saree_df[['Gender', 'Age', 'Skin Color']]
    y_color = saree_df['Saree Color']

    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    y_color_encoded = label_encoders['Gender'].transform(y_color)

    clf = RandomForestClassifier()
    clf.fit(X, y_color_encoded)
    for column in user_input:
        user_input[column] = label_encoders[column].transform([user_input[column]])[0]

    predicted_color_encoded = clf.predict(pd.DataFrame(user_input, index=[0]))
    predicted_color = label_encoders['Gender'].inverse_transform(predicted_color_encoded)[0]
    
    return predicted_color

def Kamiz_Color(user_input, df):
    kamiz_df = df[df['Comfortable Clothes'] == 'Salwar Kamiz']
    X = kamiz_df[['Gender', 'Age', 'Skin Color']]
    y_color = kamiz_df['Kamiz Color']

    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    y_color_encoded = label_encoders['Gender'].transform(y_color)

    clf = RandomForestClassifier()
    clf.fit(X, y_color_encoded)

    for column in user_input:
        user_input[column] = label_encoders[column].transform([user_input[column]])[0]

  
    predicted_color_encoded = clf.predict(pd.DataFrame(user_input, index=[0]))
    predicted_color = label_encoders['Gender'].inverse_transform(predicted_color_encoded)[0]
    
    return predicted_color



if __name__ == "__main__":
    main()
