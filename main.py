
import os
os.chdir(r"C:\Users\edson\Downloads\Blood Glucose Prediction Challange")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from lstmnetwork import lstmnetwork
from aux_function import grid_search_cv, fill_missing_with_cubic_spline

# %% parameter definition

directory='Data'
folder_train='train'
folder_test='test'
data_set = ['2018', '2020']
PHs = [30, 60, 120] # prediction horizon
n_layers = 1 # 2, 3 number of LSTM layers

# %% import train data and handle missing values

sub_path_train = os.path.join(directory,folder_train)
sub_path_test = os.path.join(directory,folder_test)
train_data = {}
train_data_filled = {}
test_data = {}
test_data_filled = {}
nan_indices_train = {}
nan_indices_test = {}

if data_set == '2018':
    patient_IDs = ['559','563','570','575','588','591'] 
elif data_set == '2020':
    patient_IDs = ['540','544','552','584','596']
    
for patient in patient_IDs:    
    file = folder_train=str(patient)+'-ws-training_processed.csv'
    df = pd.read_csv(os.path.join(sub_path_train,file))
    df_filled = df.apply(fill_missing_with_cubic_spline, axis=0)
    train_data[patient] = df
    train_data_filled[patient] = df_filled
    
    # Find the indices of NaN values
    nan_indices_train[patient] = np.where(pd.isna(df))
    
    # Plot the original and filled DataFrames with 100 samples
    fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(15, 4))
    
    for i, col in enumerate(df.columns):
        # Plot original data
        axes[i].plot(df.index[:100], df[col][:100], marker='o', color='b', label='Original')
        
        # Plot data after cubic spline interpolation
        axes[i].plot(df_filled.index[:100], df_filled[col][:100], linestyle='-', color='r', label='Interpolated')
    
        axes[i].set_title(col)
        axes[i].legend()
    
    plt.tight_layout()
    
    # Save the figure to a file
    fig.savefig(f'interpolation/interpolation_comparison_overlay_{patient}_train.png')
    
    plt.close()
    
    file = folder_test=str(patient)+'-ws-testing_processed.csv'
    df = pd.read_csv(os.path.join(sub_path_test,file))
    # df_filled = df.ffill(axis = 0)
    df_filled = df.apply(fill_missing_with_cubic_spline, axis=0)
    test_data[patient] = df
    test_data_filled[patient] = df_filled
    
    # Find the indices of NaN values
    nan_indices_test[patient] = np.where(pd.isna(df))
    
    # Plot the original and filled DataFrames with 100 samples
    fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(15, 4))
    
    for i, col in enumerate(df.columns):
        # Plot original data
        axes[i].plot(df.index[:100], df[col][:100], marker='o', color='b', label='Original')
        
        # Plot data after cubic spline interpolation
        axes[i].plot(df_filled.index[:100], df_filled[col][:100], linestyle='-', color='r', label='Interpolated')
    
        axes[i].set_title(col)
        axes[i].legend()
    
    plt.tight_layout()
    
    # Save the figure to a file 
    fig.savefig(f'interpolation/interpolation_comparison_overlay_{patient}_test.png')
    
    plt.close()


# %% TRAIN
for PH in PHs:
    for patient in patient_IDs: 
        print(f'\n\nPatient {patient}\n')
        df_train = train_data_filled[patient]
        df_test = test_data_filled[patient]
        
        target_variable = 'cbg'
        if data_set == '2018':
            features = ['cbg', 'finger', 'basal', 'hr', 'gsr', 'carbInput', 'bolus']
        elif data_set == '2020':
            features = ['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus'] #2020
    
    
        # Normalize data using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_train.loc[:, features])
        df_scaled_train = scaler.transform(df_train.loc[:, features])
        df_scaled_train = pd.DataFrame(df_scaled_train, columns=features, index=df_train.index)
        
        df_scaled_test = scaler.transform(df_test.loc[:, features])
        df_scaled_test = pd.DataFrame(df_scaled_test, columns=features, index=df_test.index)
        
        # Convert data to sequences for LSTM
        sequence_length = PH  # Adjusted for forecasting 30, 60, or 120 minutes into the future
        sequences_train = []
        target_train = []
        
        for i in range(len(df_scaled_train) - 2*sequence_length):
            seq = df_scaled_train.iloc[i:i + sequence_length][features].values
            target_val = df_scaled_train.iloc[i + 2*sequence_length][target_variable]
            sequences_train.append(seq)
            target_train.append(target_val)
           
            
        sequences_test = []
        target_test = []
        actual_target_test = [] # not scaled
        
        for i in range(len(df_scaled_test) - 2*sequence_length):
            seq = df_scaled_test.iloc[i:i + sequence_length][features].values
            target_val = df_scaled_test.iloc[i + 2*sequence_length][target_variable]
            sequences_test.append(seq)
            target_test.append(target_val)
            
            actual_target_val = df_test.iloc[i + 2*sequence_length][target_variable]
            actual_target_test.append(actual_target_val)
                
        # Convert to numpy arrays
        x_train = np.array(sequences_train)
        y_train = np.array(target_train)
        
        x_test = np.array(sequences_test)
        y_test = np.array(target_test)
        
        #Compute the correlation matrix
        correlation_matrix = df_scaled_train.corr()
        
        # Plot a heatmap of the correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.savefig(f'correlation/correlation_heatmap_{patient}.png')
    
        
        # Build model
        n_outputs = 1
        k = 5
        param_grid = {
            'activation': ['relu', 'linear']
        }
        
        # Perform grid search
        idx_best_param = grid_search_cv(n_layers,n_outputs, param_grid, k, x_train, y_train)
        activation = param_grid['activation'][idx_best_param[0]]
        # activation = 'linear'
        
        # train model
        x_train, x_val, y_train, y_val  = train_test_split(x_train,y_train, test_size = 0.15, random_state=0);
                
        if n_layers == 1:
            model = lstmnetwork.onelayer(x_train, n_outputs, activation)
        elif n_layers == 2:
            model = lstmnetwork.twolayer(x_train, n_outputs, activation)
        elif n_layers == 3:
            model = lstmnetwork.threelayer(x_train, n_outputs, activation)
        
        model = lstmnetwork.trainnet(model, x_train, y_train, x_val, y_val)
    
        lstmnetwork.testnet(patient, PH, activation, model, scaler, x_test, y_test, actual_target_test)
        
        