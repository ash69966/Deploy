import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Upload two datasets and merge them
st.header('Upload and Merge Datasets')
data_file_1 = st.file_uploader('Upload first dataset', type=['csv', 'xlsx'])
data_file_2 = st.file_uploader('Upload second dataset', type=['csv', 'xlsx'])

if data_file_1 is not None and data_file_2 is not None:
    df1 = pd.read_csv(data_file_1)
    df2 = pd.read_csv(data_file_2)
    merged_df = pd.merge(df1, df2, on='id')
    st.write('Merged Dataset')
    st.write(merged_df)

    # Check summary statistics and data types
    st.header('Summary Statistics and Data Types')
    st.write('Summary Statistics')
    st.write(merged_df.describe())
    st.write('Data Types')
    st.write(merged_df.info())

    # Check correlation
    st.header('Correlation')
    st.write(merged_df.corr())

    # Print numerical and categorical features
    st.header('Numerical and Categorical Features')
    numerical_features = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = merged_df.select_dtypes(include=['object']).columns.tolist()
    st.write('Numerical Features')
    st.write(numerical_features)
    st.write('Categorical Features')
    st.write(categorical_features)
    
    # Display columns with missing values
    null_columns = merged_df.columns[merged_df.isnull().any()]
    if len(null_columns) > 0:
        st.write("Columns with missing values:", null_columns)
    else:
        st.write("No columns with missing values found")

        # Display columns with missing values
    null_columns = merged_df.columns[merged_df.isnull().any()]
    if len(null_columns) > 0:
        st.write("Columns with missing values:", null_columns)
    else:
        st.write("No columns with missing values found")

        # Imputation based on user choice for each column separately
    imputation_methods = ['Drop NA', 'Mean Imputation', 'Median Imputation', 'Mode Imputation']
    cleaned_df = merged_df.copy()
    for i, col in enumerate(null_columns):
        st.write(f"Imputation for {col}")
        imputation_choice = st.selectbox(f'Choose an imputation method for column {i}:', imputation_methods, key=f'imputation_{i}')
        if imputation_choice == 'Drop NA':
            cleaned_df = cleaned_df.dropna(subset=[col])
        elif imputation_choice == 'Mean Imputation':
            imputer = SimpleImputer(strategy='mean')
            cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]])
        elif imputation_choice == 'Median Imputation':
            imputer = SimpleImputer(strategy='median')
            cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]])
        elif imputation_choice == 'Mode Imputation':
            imputer = SimpleImputer(strategy='most_frequent')
            cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]])
    
    st.write('Cleaned Dataset')
    st.write(cleaned_df)

   
        # Preprocessing
    st.header('Preprocessing')
    preprocessing_methods = ['One-Hot Encoding', 'Label Encoding']
    preprocessing_choice = st.selectbox('Choose a preprocessing method:', preprocessing_methods)
    if preprocessing_choice == 'One-Hot Encoding':
        cat_features = [col for col in cleaned_df.columns if cleaned_df[col].dtype != 'float64']
        num_features = [col for col in cleaned_df.columns if cleaned_df[col].dtype == 'float64']
        encoder = OneHotEncoder()
        encoded_cat = pd.DataFrame(encoder.fit_transform(cleaned_df[cat_features]).toarray(), columns=encoder.get_feature_names(cat_features))
        encoded_df = pd.concat([encoded_cat, cleaned_df[num_features]], axis=1)
    elif preprocessing_choice == 'Label Encoding':
        encoder = LabelEncoder()
        encoded_df = cleaned_df.apply(encoder.fit_transform)

    st.write('Encoded Dataset')
    st.write(encoded_df)

    
        # Train-test split and model training
    st.header('Model Training')

        # Select target variable to drop
    target_variable = st.selectbox('Select target variable:', encoded_df.columns)

    # Split the data into training and testing sets
    X = encoded_df.drop(target_variable, axis=1)
    y = encoded_df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the hyperparameter grid
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean']}

    # Create a KNN regressor object
    knn = KNeighborsRegressor()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Train the model with the best hyperparameters
    best_knn = grid_search.best_estimator_
    best_knn.fit(X_train, y_train)
    st.write('Best k value:',best_knn)

    # Model evaluation
    st.header('Model Evaluation')
    st.write(f'Training Score: {best_knn.score(X_train, y_train)}')
    st.write(f'Test Score: {best_knn.score(X_test, y_test)}')

    # Get the predictions for training and testing data
    train_predictions = best_knn.predict(X_train)
    test_predictions = best_knn.predict(X_test)

    # Calculate RMSE, MSE, and R2 score
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    st.write(f'Training RMSE: {train_rmse}')
    st.write(f'Test RMSE: {test_rmse}')

    st.write(f'Training MSE: {train_mse}')
    st.write(f'Test MSE: {test_mse}')

    st.write(f'Training R2 Score: {train_r2}')
    st.write(f'Test R2 Score: {test_r2}')