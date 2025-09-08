import pandas as pd
import numpy as np
import re

# Load the dataset
# Make sure to replace 'mental_health_dataset.csv' with the exact filename
try:
    df = pd.read_csv('mental_health_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'mental_health_dataset.csv' was not found. Please check the file path and name.")
    exit()

# Display the first few rows and column information
print("\n--- Initial Data Overview ---")
print(df.head())
print("\n--- Column Information ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Handle missing numerical values
# For example, let's assume 'Sleep_Hours' is a numerical column
if 'Sleep_Hours' in df.columns:
    df.fillna(df.median(), inplace=True)

# Handle missing categorical values
# Assuming 'Occupation' is a categorical column
if 'Occupation' in df.columns:
    df['Occupation'].fillna('Unknown', inplace=True)

# Note: The dataset has various columns. You'll need to apply this
# to all relevant columns with missing data.
    
# Normalize categorical data (e.g., 'Gender' column)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.lower().str.strip()
    df['Gender'].replace({'f': 'female', 'm': 'male', 'non-binary': 'non_binary'}, inplace=True)

# Cleaning text-based columns
# For instance, if there's a column with free-form text, you'd apply NLP preprocessing
# The dataset has text-derived features, so raw text might not be available.
# But if it is, here's a general approach:
if 'text_column' in df.columns: # Replace 'text_column' with the actual column name
    df['text_column'] = df['text_column'].str.lower()
    # Remove punctuation and special characters
    df['text_column'] = df['text_column'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

# Ensure sentiment scores are of the correct numerical type
# Assuming sentiment scores columns exist, based on research
sentiment_cols = ['sentiment_compound', 'negative', 'neutral', 'positive']
for col in sentiment_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(0, inplace=True)

# Convert 'Time' column to datetime objects
# Assuming a 'Time' or 'Timestamp' column exists
if 'Time' in df.columns:
    df = pd.to_datetime(df, errors='coerce')

# Drop any columns that are not relevant to your research question
# For example, if 'User_ID' is just an index
if 'User_ID' in df.columns:
    df.drop('User_ID', axis=1, inplace=True)

# Remove duplicate entries
df.drop_duplicates(inplace=True)
print("\n--- Dataset after cleaning ---")
print(df.info())