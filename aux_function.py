
import numpy as np
from scipy.interpolate import CubicSpline 
from lstmnetwork import lstmnetwork
from sklearn.model_selection import KFold

def fill_missing_with_cubic_spline(series):
    '''Function to fill missing values using cubic spline interpolation'''
    mask = series.notna()
    
    if sum(mask) < 2:
        # If there are fewer than two non-missing values, return the original series
        return series
    
    x = np.arange(len(series))
    
    # Use only non-missing values for interpolation
    cs = CubicSpline(x[mask], series[mask], bc_type='natural')
    
    # Fill missing values using the cubic spline interpolation
    series[~mask] = cs(x[~mask])
    
    return series

def grid_search_cv(n_layers,n_outputs,param_grid,k,x,y):
    '''5-fold cross validation'''
    
    print('\nGrid search...')
    
    activation_mae = []
    for activation in param_grid['activation']:
        kf = KFold(n_splits=k)
        kf.get_n_splits(x)
        KFold(n_splits=k, random_state=0, shuffle=True)
        fold_mae = []
        for train_index, val_index in kf.split(x):
              x_train, x_val = x[train_index], x[val_index]
              y_train, y_val = y[train_index], y[val_index]
          
              # Create and train the model
              if n_layers == 1:
                  model = lstmnetwork.onelayer(x_train, n_outputs, activation)
              elif n_layers == 2:
                  model = lstmnetwork.twolayer(x_train, n_outputs, activation)
              elif n_layers == 3:
                  model = lstmnetwork.threelayer(x_train, n_outputs, activation)
                  
              model = lstmnetwork.trainnet(model, x_train, y_train, x_val, y_val)
              
              # Evaluate the model on the validation set
              metrics = lstmnetwork.validatenet(model, x_val, y_val)
              mae = metrics[2]
              
              fold_mae.append(mae)
              
        activation_mae.append(np.mean(fold_mae))
                              

    best_mae = min(activation_mae)
    idx_best = np.where(activation_mae == best_mae)[0]
    
    print('\nGrid search completed')
        
    return idx_best

                                  