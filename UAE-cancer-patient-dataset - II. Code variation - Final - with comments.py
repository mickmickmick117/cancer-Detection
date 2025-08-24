# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:04:45 2025

@author: miroz
"""




import pandas as pd # Import the pandas library for data manipulation and analysis
# import numpy as np # Import the numpy library for numerical operations
import tensorflow as tf # Import the TensorFlow library for machine learning
from sklearn.model_selection import train_test_split # Import the train_test_split function to split data
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import StandardScaler for numerical
# feature scaling and OneHotEncoder for categorical feature encoding
from sklearn.compose import ColumnTransformer # Import ColumnTransformer to apply different transformations to different columns
# from sklearn.pipeline import Pipeline # Import Pipeline to chain multiple data transformations and a model
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced visualizations
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score # Import various metrics for model evaluation

# 1. Load the dataset
file_path = r"C:\Users\miroz\OneDrive\Documents\Miro\Miro\Python\Neural Networks\NN Project\_cancer_dataset_uae.csv" # Define the path to the CSV file
try:
    df = pd.read_csv(file_path) # Read the CSV file into a pandas DataFrame
    print("Dataset loaded successfully.") # Print a success message
    print(df.head()) # Print the first few rows of the DataFrame
except FileNotFoundError:
    print(f"Error: File not found at {file_path}") # Print an error message if the file is not found
    exit() # Exit the script

# Inspect unique values in 'Outcome' column
if 'Outcome' in df.columns: # Check if the 'Outcome' column exists in the DataFrame
    print("\nUnique values in 'Outcome' column:", df['Outcome'].unique()) # Print the unique values in the 'Outcome' column
    outcome_mapping = {'Recovered': 0, 'Deceased': 1, 'Under Treatment': -1} # Define a mapping for the 'Outcome' categories to numerical values
    df['Outcome_Numerical'] = df['Outcome'].map(outcome_mapping) # Create a new column 'Outcome_Numerical' by mapping the 'Outcome' values
    df_cleaned = df[df['Outcome_Numerical'] != -1].dropna(subset=['Outcome_Numerical']) # Create a new DataFrame by removing rows where 'Outcome_Numerical' is -1 (Under Treatment) and dropping rows with NaN in 'Outcome_Numerical'
    y = df_cleaned['Outcome_Numerical'].astype(int) # Assign the 'Outcome_Numerical' column as the target variable (y) and convert it to integers
    X_original = df_cleaned.drop(['Outcome', 'Outcome_Numerical', 'Patient_ID', 'Diagnosis_Date',
                                   'Treatment_Start_Date', 'Death_Date', 'Cause_of_Death'], axis=1) # Assign the remaining columns as the feature matrix (X_original), dropping irrelevant columns
else:
    print("Error: 'Outcome' column not found in the dataset.") # Print an error message if the 'Outcome' column is missing
    exit() # Exit the script

# 3. Identify categorical and numerical features
categorical_features = ['Gender', 'Nationality', 'Emirate', 'Cancer_Type', 'Cancer_Stage',
                        'Treatment_Type', 'Hospital', 'Primary_Physician', 'Smoking_Status',
                        'Comorbidities', 'Ethnicity'] # Define a list of categorical feature names
numerical_features = ['Age', 'Weight', 'Height'] # Define a list of numerical feature names

existing_categorical = [col for col in categorical_features if col in X_original.columns] # Identify the categorical features that actually exist in the feature matrix
existing_numerical = [col for col in numerical_features if col in X_original.columns] # Identify the numerical features that actually exist in the feature matrix

print(f"\nCategorical features found: {existing_categorical}") # Print the found categorical features
print(f"Numerical features found: {existing_numerical}") # Print the found numerical features

# 4. Create preprocessing pipelines
numerical_transformer = StandardScaler() # Initialize a StandardScaler object for numerical feature scaling
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Initialize a OneHotEncoder object for categorical feature encoding, ignoring unknown values

preprocessor = ColumnTransformer( # Initialize a ColumnTransformer to apply different transformations to different columns
    transformers=[
        ('num', numerical_transformer, existing_numerical), # Apply StandardScaler to the existing numerical features
        ('cat', categorical_transformer, existing_categorical)]) # Apply OneHotEncoder to the existing categorical features

# 6. Split data
X_train_original, X_test_original, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42, stratify=y) # Split the data into training and testing sets, with 20% for testing, a random seed for reproducibility, and stratified sampling based on the target variable

# 7. Preprocess the training and testing data
X_train_processed = preprocessor.fit_transform(X_train_original) # Fit the preprocessor on the training data and transform it
X_test_processed = preprocessor.transform(X_test_original) # Transform the testing data using the fitted preprocessor

# 8. Define a more complex neural network model
model = tf.keras.Sequential([ # Define a sequential neural network model
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)), # Add a dense layer with 128 neurons, ReLU activation, and the input shape matching the number of features
    tf.keras.layers.BatchNormalization(), # Add batch normalization layer
    tf.keras.layers.Dense(64, activation='relu'), # Add another dense layer with 64 neurons and ReLU activation
    tf.keras.layers.BatchNormalization(), # Add batch normalization layer
    tf.keras.layers.Dense(32, activation='relu'), # Add another dense layer with 32 neurons and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid') # Add the output dense layer with 1 neuron and sigmoid activation for binary classification
])

# 9. Compile the model
model.compile(optimizer='adam', # Use the Adam optimizer for training
              loss='binary_crossentropy', # Use binary cross-entropy as the loss function for binary classification
              metrics=['accuracy', 'Precision', 'Recall']) # Track accuracy, precision, and recall during training and evaluation

# 10. Train the model
history = model.fit(X_train_processed, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0) # Train the model on the processed training data for 30 epochs, with a batch size of 32, using 10% of the training data for validation, and without verbose output

# --- Training History Plots ---
plt.figure(figsize=(16, 4)) # Create a figure for the plots with a specified size
plt.subplot(1, 3, 1) # Create the first subplot (1 row, 3 columns, first plot)
plt.plot(history.history['accuracy'], label='Train Accuracy') # Plot the training accuracy over epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Plot the validation accuracy over epochs
plt.title('Model Accuracy') # Set the title of the subplot
plt.xlabel('Epoch') # Set the x-axis label
plt.ylabel('Accuracy') # Set the y-axis label
plt.legend() # Display the legend

plt.subplot(1, 3, 2) # Create the second subplot
plt.plot(history.history['loss'], label='Train Loss') # Plot the training loss over epochs
plt.plot(history.history['val_loss'], label='Validation Loss') # Plot the validation loss over epochs
plt.title('Model Loss') # Set the title of the subplot
plt.xlabel('Epoch') # Set the x-axis label
plt.ylabel('Loss') # Set the y-axis label
plt.legend() # Display the legend

plt.subplot(1, 3, 3) # Create the third subplot
plt.plot(history.history['Precision'], label='Train Precision') # Plot the training precision over epochs
plt.plot(history.history['val_Precision'], label='Validation Precision') # Plot the validation precision over epochs
plt.plot(history.history['Recall'], label='Train Recall') # Plot the training recall over epochs
plt.plot(history.history['val_Recall'], label='Validation Recall') # Plot the validation recall over epochs
plt.title('Precision and Recall') # Set the title of the subplot
plt.xlabel('Epoch') # Set the x-axis label
plt.ylabel('Value') # Set the y-axis label
plt.legend() # Display the legend

plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show() # Display the plots

# 11. Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test_processed, y_test, verbose=0) # Evaluate the model on the processed test data and get the loss, accuracy, precision, and recall
print(f"\nTest Loss: {loss:.4f}") # Print the test loss
print(f"\nTest Accuracy: {accuracy:.4f}") # Print the test accuracy
print(f"\nTest Precision: {precision:.4f}") # Print the test precision
print(f"\nTest Recall: {recall:.4f}") # Print the test recall

# --- Evaluation Plots ---
y_pred_prob = model.predict(X_test_processed).flatten() # Get the probability predictions for the test data and flatten the array
y_pred = (y_pred_prob > 0.5).astype(int) # Convert the probability predictions to binary predictions (0 or 1) based on a threshold of 0.5

print("\nClassification Report:") # Print the classification report
print(classification_report(y_test, y_pred)) # Generate and print the classification report (precision, recall, F1-score, support)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # Calculate the confusion matrix
plt.figure(figsize=(6, 5)) # Create a figure for the confusion matrix plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # Create a heatmap of the confusion matrix with annotations, integer formatting, and the Blues colormap
            xticklabels=['Recovered', 'Deceased'], yticklabels=['Recovered', 'Deceased']) # Set the labels for the x and y axes
plt.title('Confusion Matrix') # Set the title of the plot
plt.xlabel('Predicted') # Set the x-axis label
plt.ylabel('Actual') # Set the y-axis label
plt.show() # Display the plot

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
roc_auc = auc(fpr, tpr) # Calculate the area under the ROC curve (AUC)
plt.figure(figsize=(8, 6)) # Create a figure for the ROC curve plot
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})') # Plot the ROC curve
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Plot the diagonal line representing a random classifier
plt.xlabel('False Positive Rate') # Set the x-axis label
plt.ylabel('True Positive Rate') # Set the y-axis label
plt.title('Receiver Operating Characteristic (ROC)') # Set the title of the plot
plt.legend(loc='lower right') # Display the legend
plt.show() # Display the plot

# Precision-Recall Curve
precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_test, y_pred_prob) # Calculate the precision, recall, and thresholds for the precision-recall curve
avg_precision = average_precision_score(y_test, y_pred_prob) # Calculate the average precision
plt.figure(figsize=(8, 6)) # Create a figure for the precision-recall curve plot
plt.plot(recall_pr, precision_pr, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})') # Plot the precision-recall curve
plt.xlabel('Recall') # Set the x-axis label
plt.ylabel('Precision') # Set the y-axis label
plt.title('Precision-Recall Curve') # Set the title of the plot
plt.legend(loc='lower left') # Display the legend
plt.grid(True) # Add a grid to the plot
plt.show() # Display the plot

# --- Plotting a different feature (e.g., Age Distribution by Outcome) ---
plt.figure(figsize=(8, 6)) # Create a figure for the box plot
sns.boxplot(x='Outcome', y='Age', data=df_cleaned) # Create a box plot of Age distribution by Outcome
plt.title('Age Distribution by Outcome') # Set the title of the plot
plt.xlabel('Outcome (0: Recovered, 1: Deceased)') # Set the x-axis label
plt.ylabel('Age') # Set the y-axis label
plt.xticks([0, 1], ['Recovered', 'Deceased']) # Set the x-axis tick labels
plt.show() # Display the plot

# --- Plotting Cancer Type Distribution ---
if 'Cancer_Type' in df_cleaned.columns: # Check if the 'Cancer_Type' column exists
    plt.figure(figsize=(10, 6)) # Create a figure for the count plot
    sns.countplot(y='Cancer_Type', data=df_cleaned, order=df_cleaned['Cancer_Type'].value_counts().index) # Create a count plot of Cancer Types, ordered by frequency
    plt.title('Distribution of Cancer Types') # Set the title of the plot
    plt.xlabel('Count') # Set the x-axis label
    plt.ylabel('Cancer Type') # Set the y-axis label
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

# --- Pie Chart of Cancer Type Frequencies (%) ---
if 'Cancer_Type' in df_cleaned.columns: # Check if the 'Cancer_Type' column exists
    cancer_type_counts = df_cleaned['Cancer_Type'].value_counts() # Get the counts of each Cancer Type
    plt.figure(figsize=(8, 8)) # Create a figure for the pie chart
    plt.pie(cancer_type_counts, labels=cancer_type_counts.index, autopct='%1.1f%%', startangle=140) # Create a pie chart of Cancer Type frequencies with percentage labels
    plt.title('Frequency of Different Cancer Types (%)') # Set the title of the plot
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

# --- Distribution of Top 3 Cancer Types by Gender ---
if 'Cancer_Type' in df_cleaned.columns and 'Gender' in df_cleaned.columns: # Check if both 'Cancer_Type' and 'Gender' columns exist
    # Get counts of each cancer type
    cancer_type_counts = df_cleaned['Cancer_Type'].value_counts() # Get the counts of each Cancer Type

    # Get the top 3 cancer types
    top_3_cancer_types = cancer_type_counts.head(3).index # Get the index (names) of the top 3 most frequent cancer types

    # Filter the data to include only the top 3 cancer types
    df_top3 = df_cleaned[df_cleaned['Cancer_Type'].isin(top_3_cancer_types)] # Create a DataFrame containing only the rows where 'Cancer_Type' is one of the top 3

    # Create a countplot of the top 3 cancer types by gender
    plt.figure(figsize=(10, 6)) # Create a figure for the count plot
    sns.countplot(x='Cancer_Type', hue='Gender', data=df_top3) # Create a count plot of the top 3 Cancer Types, with hue based on Gender
    plt.title('Distribution of Top 3 Cancer Types by Gender') # Set the title of the plot
    plt.xlabel('Cancer Type') # Set the x-axis label
    plt.ylabel('Count') # Set the y-axis label
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

# --- Distribution of Top 3 Cancer Types by Smoking Status ---
if 'Cancer_Type' in df_cleaned.columns and 'Smoking_Status' in df_cleaned.columns: # Check if both 'Cancer_Type' and 'Smoking_Status' columns exist
    # Get counts of each cancer type
    cancer_type_counts = df_cleaned['Cancer_Type'].value_counts() # Get the counts of each Cancer Type

    # Get the top 3 cancer types
    top_3_cancer_types = cancer_type_counts.head(3).index # Get the index (names) of the top 3 most frequent cancer types

    # Filter the data to include only the top 3 cancer types
    df_top3 = df_cleaned[df_cleaned['Cancer_Type'].isin(top_3_cancer_types)] # Create a DataFrame containing only the rows where 'Cancer_Type' is one of the top 3

    # Create a countplot of the top 3 cancer types by smoking status
    plt.figure(figsize=(10, 6)) # Create a figure for the count plot
    sns.countplot(x='Cancer_Type', hue='Smoking_Status', data=df_top3) #