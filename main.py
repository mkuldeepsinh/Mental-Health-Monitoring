# # ----------------------------
# # Install dependencies (run in terminal if not installed)
# # ----------------------------
# # pip install scikit-learn pandas numpy

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.utils import class_weight
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # ----------------------------
# # Load dataset (local file)
# # ----------------------------
# csv_file = "mental_health_dataset.csv"
# df = pd.read_csv(csv_file)
# print(" Dataset loaded successfully!")
# print(df.head())

# # ----------------------------
# # Data Cleaning
# # ----------------------------
# if 'Timestamp' in df.columns:
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df.dropna(subset=['Timestamp'], inplace=True)
#     df.sort_values(by='Timestamp', inplace=True)

# if 'Gender' in df.columns:
#     df['Gender'] = df['Gender'].str.lower().str.strip()
#     df['Gender'] = df['Gender'].replace({
#         'f': 'female', 'm': 'male', 'non-binary': 'non_binary'
#     })

# # Fill missing categorical with "Unknown"
# for col in df.select_dtypes(include='object').columns:
#     df[col] = df[col].fillna('Unknown')

# # Fill missing numeric with median
# for col in df.select_dtypes(include=np.number).columns:
#     df[col] = df[col].fillna(df[col].median())

# # Drop duplicates
# df.drop_duplicates(inplace=True)
# print("\n--- Dataset after cleaning ---")
# print(df.info())

# # ----------------------------
# # Features & Target
# # ----------------------------
# y = df['treatment'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
# X = df.drop(columns=['Timestamp', 'treatment'], errors='ignore')

# numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = X.select_dtypes(include='object').columns.tolist()

# # ----------------------------
# # Preprocessing
# # ----------------------------
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
#     ]
# )

# X_processed = preprocessor.fit_transform(X)
# print(f"\n Shape after preprocessing: {X_processed.shape}")

# # ----------------------------
# # Train-test split
# # ----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_processed, y.values, test_size=0.2, random_state=42, stratify=y.values
# )

# # ----------------------------
# # Handle class imbalance
# # ----------------------------
# class_weights_array = class_weight.compute_class_weight(
#     'balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict = dict(enumerate(class_weights_array))
# print("\nClass Weights:", class_weights_dict)

# # ----------------------------
# # Random Forest with Hyperparameter Tuning
# # ----------------------------
# rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# param_dist = {
#     'n_estimators': [200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# search = RandomizedSearchCV(
#     rf, param_distributions=param_dist,
#     n_iter=20, scoring='accuracy',
#     cv=3, verbose=2, random_state=42, n_jobs=-1
# )

# search.fit(X_train, y_train)
# best_rf = search.best_estimator_
# print("\n Best Parameters:", search.best_params_)

# # ----------------------------
# # Evaluation
# # ----------------------------
# y_pred = best_rf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print(f"\n Random Forest Test Accuracy: {acc:.4f}")

# print("\n--- Classification Report ---")
# print(classification_report(y_test, y_pred))

# print("\n--- Confusion Matrix ---")
# print(confusion_matrix(y_test, y_pred))

# ----------------------------
# Install dependencies (if not installed)
# ----------------------------
# pip install scikit-learn pandas numpy xgboost

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb # Import XGBoost

# ----------------------------
# Load dataset (local file)
# ----------------------------
csv_file = "mental_health_dataset.csv"
df = pd.read_csv(csv_file)
print(" Dataset loaded successfully!")

# ----------------------------
# 1. Advanced Data Cleaning & Feature Engineering
# ----------------------------

# Drop Timestamp as it's not needed for prediction
if 'Timestamp' in df.columns:
    df = df.drop('Timestamp', axis=1)

# Standardize Gender column
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.lower().str.strip()
    # Comprehensive list of variations to map to a few categories
    gender_map = {
        'f': 'female', 'woman': 'female', 'female': 'female',
        'm': 'male', 'man': 'male', 'male': 'male',
        'non-binary': 'non_binary'
    }
    df['Gender'] = df['Gender'].map(gender_map).fillna('other')

# Handle High-Cardinality 'Country' feature by grouping rare ones
# Any country with fewer than 10 occurrences will be labeled 'Other'
country_counts = df['Country'].value_counts()
rare_countries = country_counts[country_counts < 10].index
df['Country'] = df['Country'].replace(rare_countries, 'Other')
print("\nTop 5 Country Counts after grouping:")
print(df['Country'].value_counts().head())


# ** FEATURE ENGINEERING: Create a 'symptom_score' **
# Summing up binary/categorical indicators of mental health struggles
symptom_cols = [
    'family_history', 'Growing_Stress', 'Changes_Habits',
    'Mental_Health_History', 'Coping_Struggles', 'Work_Interest',
    'Social_Weakness'
]
for col in symptom_cols:
    # Convert 'Yes' to 1, 'No' to 0, and others ('Maybe', 'Don't know') to 0.5
    df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0.5)

df['symptom_score'] = df[symptom_cols].sum(axis=1)
print("\n Created 'symptom_score' feature.")


# Fill remaining missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('Unknown')
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

df.drop_duplicates(inplace=True)
print("\n--- Dataset after cleaning and feature engineering ---")
print(df.info())


# ----------------------------
# Features & Target
# ----------------------------
y = df['treatment'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
X = df.drop(columns=['treatment'])

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns
)

X_processed = preprocessor.fit_transform(X)
print(f"\n Shape after preprocessing: {X_processed.shape}")

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y.values, test_size=0.2, random_state=42, stratify=y.values
)

# ----------------------------
# 2. Handle Class Imbalance for XGBoost
# ----------------------------
# Calculate scale_pos_weight for XGBoost
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")


# ----------------------------
# 3. XGBoost with Hyperparameter Tuning
# ----------------------------
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight, # Use the calculated weight
    random_state=42,
    n_jobs=-1
)

# A more focused parameter distribution for XGBoost
param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_dist,
    n_iter=30, # Increased iterations for better search
    scoring='accuracy',
    cv=5, # Increased folds for more robust validation
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Starting XGBoost hyperparameter search...")
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print("\n Best Parameters:", search.best_params_)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = best_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🚀 XGBoost Test Accuracy: {acc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Treatment', 'Treatment']))

print("\n--- Confusion Matrix ---")
# 
print(confusion_matrix(y_test, y_pred))

# got 81% accuracy by using xgboost

# ----------------------------
# Save the model
# ----------------------------
# joblib.dump(best_xgb, 'best_xgb_model.joblib')