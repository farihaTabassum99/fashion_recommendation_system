

# All needed libraries
import pandas as pd
import numpy as np
import random
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

df.drop(columns=['Unused','Timestamp','Email Address','Score','Name'], inplace=True)

# Remove Bangla characters
df = df.apply(lambda x: x.str.split('(', expand=True)[0])

# Filter the DataFrame to remove rows where 'Comfortable Clothes' count is less than 5
df = df[df.groupby('Comfortable Clothes')['Comfortable Clothes'].transform('count') >= 5]

# Handle missing value 'Gender' column
df['Gender'] = df['Gender'].str.strip()
df['Gender'].fillna('Female',inplace=True)
df['Gender'].value_counts()

# Replace values in the 'Age' column
df['Age'].replace({'12-22': '12-23'}, inplace=True)

# Define gender-specific color preferences
male_colors = {'Panjabi': ['Gray', 'Blue', 'White'], 'Shirt': ['Black', 'Blue', 'White']}
female_colors = {'Saree': ['White', 'Blue', 'Black'], 'Kamiz': ['Green', 'Red', 'Black']}


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

df[['Punjabi Color', 'Shirt Color', 'Saree Color', 'Kamiz Color']] = df.apply(fill_null_color, axis=1)

X = df[['Gender', 'Age', 'Skin Color']]
y_clothes = df['Comfortable Clothes']


X_train, X_test, y_clothes_train, y_clothes_test = train_test_split(X, y_clothes, test_size=0.2, random_state=42)


label_encoder = LabelEncoder()

y_clothes_train_encoded = label_encoder.fit_transform(y_clothes_train)
y_clothes_test_encoded = label_encoder.transform(y_clothes_test) 

encoder = ce.TargetEncoder(cols=['Gender', 'Age', 'Skin Color'])


X_train_encoded = encoder.fit_transform(X_train, y_clothes_train_encoded)

X_test_encoded = encoder.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train_encoded, y_clothes_train_encoded)

y_clothes_pred = clf.predict(X_test_encoded)

knn = KNeighborsClassifier()
knn.fit(X_train_encoded, y_clothes_train_encoded)

dt = DecisionTreeClassifier()
dt.fit(X_train_encoded, y_clothes_train_encoded)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train_encoded)

y_clothes_pred_knn = knn.predict(X_test_encoded)

y_clothes_pred_dt = dt.predict(X_test_encoded)

cluster_pred = kmeans.predict(X_test_encoded)

classifiers = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Means Clustering": KMeans(n_clusters=2)
}

accuracies = []
labels = []

for label, clf in classifiers.items():
    if label != "K-Means Clustering":
        clf.fit(X_train_encoded, y_clothes_train_encoded)
        y_pred = clf.predict(X_test_encoded)
        accuracy = accuracy_score(y_clothes_test_encoded, y_pred)
        accuracies.append(accuracy)
        labels.append(label)
    else:
        clf.fit(X_train_encoded)
        score = silhouette_score(X_train_encoded, clf.labels_)
        accuracies.append(score)
        labels.append(label + " Silhouette Score")

X = df[['Gender', 'Age', 'Skin Color']].copy()
y_clothes = df['Comfortable Clothes']

label_encoder_input = LabelEncoder()
X['Gender'] = label_encoder_input.fit_transform(X['Gender'])

X['Age'] = pd.to_numeric(X['Age'], errors='coerce')

age_bins = [0, 22, 40, 55, float('inf')]
age_labels = ['12-22', '23-40', '41-55', '55+']

X['Age'] = pd.cut(X['Age'], bins=age_bins, labels=age_labels, right=False)

label_encoder_age = LabelEncoder()
X['Age'] = label_encoder_age.fit_transform(X['Age'])

label_encoder_skin_color = LabelEncoder()
X['Skin Color'] = label_encoder_skin_color.fit_transform(X['Skin Color'])

label_encoder_clothes = LabelEncoder()
y_clothes_encoded = label_encoder_clothes.fit_transform(y_clothes)

X_train_clothes, X_test_clothes, y_clothes_train, y_clothes_test = train_test_split(X, y_clothes_encoded, test_size=0.2, random_state=42)

knn_classifier_clothes = KNeighborsClassifier()
knn_classifier_clothes.fit(X_train_clothes, y_clothes_train)

def Punjabi_Color(user_input, df):
    Punjabi_df = df[df['Comfortable Clothes'] == 'Panjabi']
    
    if Punjabi_df.empty:
        return "No data available for Punjabi category"

    X = Punjabi_df[['Gender', 'Age', 'Skin Color']]
    y_color = Punjabi_df['Punjabi Color']

    label_encoders = {}
    for column in ['Gender', 'Age', 'Skin Color']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    label_encoder_color = LabelEncoder()
    y_color_encoded = label_encoder_color.fit_transform(y_color)

    knn_classifier_color = KNeighborsClassifier()
    knn_classifier_color.fit(X, y_color_encoded)

    input_df = pd.DataFrame(user_input, index=[0])

    for column in ['Gender', 'Age', 'Skin Color']:
        input_df[column] = label_encoders[column].transform([user_input[column]])

    color_prediction = knn_classifier_color.predict(input_df)

    predicted_color = label_encoder_color.inverse_transform(color_prediction)[0]
    return predicted_color



def Shirt_Color(user_input, df):
    Shirt_df = df[df['Comfortable Clothes'] == 'Shirt']
    X = Shirt_df[['Gender', 'Age', 'Skin Color']]
    y_color = Shirt_df['Shirt Color']

    label_encoders = {}
    for column in ['Gender', 'Age', 'Skin Color']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    label_encoder_color = LabelEncoder()
    y_color_encoded = label_encoder_color.fit_transform(y_color)

    knn_classifier_color = KNeighborsClassifier()
    knn_classifier_color.fit(X, y_color_encoded)

    input_df = pd.DataFrame(user_input, index=[0])

    for column in ['Gender', 'Age', 'Skin Color']:
        input_df[column] = label_encoders[column].transform([user_input[column]])

    color_prediction = knn_classifier_color.predict(input_df)
    predicted_color = label_encoder_color.inverse_transform(color_prediction)[0]
    return predicted_color


def Kamiz_Color(user_input, df):
    kamiz_df = df[df['Comfortable Clothes'] == 'Salwar Kamiz']
    X = kamiz_df[['Gender', 'Age', 'Skin Color']]
    y_color = kamiz_df['Kamiz Color']

    label_encoders = {}
    for column in ['Gender', 'Age', 'Skin Color']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    label_encoder_color = LabelEncoder()
    y_color_encoded = label_encoder_color.fit_transform(y_color)

    knn_classifier_color = KNeighborsClassifier()
    knn_classifier_color.fit(X, y_color_encoded)
    input_df = pd.DataFrame(user_input, index=[0])

    for column in ['Gender', 'Age', 'Skin Color']:
        input_df[column] = label_encoders[column].transform([user_input[column]])

    color_prediction = knn_classifier_color.predict(input_df)

    predicted_color = label_encoder_color.inverse_transform(color_prediction)[0]
    return predicted_color


def Saree_Color(user_input, df):
    saree_df = df[df['Comfortable Clothes'] == 'Saree']
    X = saree_df[['Gender', 'Age', 'Skin Color']]
    y_color = saree_df['Saree Color']

    label_encoders = {}
    for column in ['Gender', 'Age', 'Skin Color']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    label_encoder_color = LabelEncoder()
    y_color_encoded = label_encoder_color.fit_transform(y_color)

    knn_classifier_color = KNeighborsClassifier()
    knn_classifier_color.fit(X, y_color_encoded)
    input_df = pd.DataFrame(user_input, index=[0])

    for column in ['Gender', 'Age', 'Skin Color']:
        input_df[column] = label_encoders[column].transform([user_input[column]])

    color_prediction = knn_classifier_color.predict(input_df)
    predicted_color = label_encoder_color.inverse_transform(color_prediction)[0]
    return predicted_color


def main():
    # Streamlit app
    st.title("Comfortable Clothes and Color Prediction")

    user_input = {}
    user_input['Gender'] = st.selectbox("Select Gender", df['Gender'].unique())
    user_input['Age'] = st.selectbox('Age', df['Age'].unique())
    user_input['Skin Color'] = st.selectbox("Select Skin Color", df['Skin Color'].unique())

    def Comfortable_Clothes(user_input, df):
        X = df[['Gender', 'Age', 'Skin Color']]
        y_clothes = df['Comfortable Clothes']

        label_encoders = {}
        for column in ['Gender', 'Age', 'Skin Color']:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])

        label_encoder_clothes = LabelEncoder()
        y_clothes_encoded = label_encoder_clothes.fit_transform(y_clothes)

        knn_classifier_clothes = KNeighborsClassifier()
        knn_classifier_clothes.fit(X, y_clothes_encoded)

        input_df = pd.DataFrame(user_input, index=[0])

        for column in ['Gender', 'Age', 'Skin Color']:
            input_df[column] = label_encoders[column].transform([user_input[column]])

        clothes_prediction = knn_classifier_clothes.predict(input_df)
        predicted_cloth = label_encoder_clothes.inverse_transform(clothes_prediction)[0]
        return predicted_cloth

    def color_predict(predicted_cloth):
        if predicted_cloth == 'Panjabi':
            predicted_color = Punjabi_Color(user_input, df)
        elif predicted_cloth == 'Shirt':
            predicted_color = Shirt_Color(user_input, df)
        elif predicted_cloth == 'Saree':
            predicted_color = Saree_Color(user_input, df)
        elif predicted_cloth == 'Salwar Kamiz':
            predicted_color = Kamiz_Color(user_input, df)
        else:
            predicted_color = "No color prediction function available for the predicted comfortable clothes."
        return predicted_color

    predicted_cloth = ""
    predicted_color = ""

    if st.button("**Predict**"):
        predicted_cloth = Comfortable_Clothes(user_input, df)
        predicted_color = color_predict(predicted_cloth)

    st.subheader("Predictions")
    st.write(f"Predicted Comfortable Clothes: **{predicted_cloth}**")
    st.write(f"Predicted Color: **{predicted_color}**")


if __name__ == "__main__":
    main()
