
from keras.layers import Dense,Dropout, SimpleRNN, GRU,LSTM, Masking, Bidirectional,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, R_squared, correlation_coefficient, FIT, MARD

class lstmnetwork:    
    def onelayer(X_train, n_outputs, activation):
       
        #Model building
        dropout = 0.1
        recurrent_dropout = 0.2
    
        model = Sequential()
        model.add(LSTM(10,
                    return_sequences=False,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful=False,
                    input_shape = X_train[0].shape
                    #batch_size=batch_size
                ))
        model.add(Dense(n_outputs, activation=activation))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        model.summary()
    
        return model
    
    def twolayer(X_train, n_outputs, activation):
       
        #Model building
        dropout = 0.1
        recurrent_dropout = 0.2
    
        model = Sequential()
        model.add(Bidirectional(LSTM(24,
                    return_sequences=True,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful=False),
                    input_shape = X_train[0].shape
                    #batch_size=batch_size
                ))
        model.add(LSTM(12,
                    return_sequences=False,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful = False))
        model.add(Dense(n_outputs, activation=activation))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        model.summary()
    
        return model
    
    def threelayer(X_train, n_outputs, activation):
       
        #Model building
        dropout = 0.1
        recurrent_dropout = 0.2
    
        model = Sequential()
        model.add(Bidirectional(LSTM(128,
                    return_sequences=True,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful=False),
                    input_shape = X_train[0].shape
                    #batch_size=batch_size
                ))
        model.add(LSTM(64, return_sequences=True,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful=False,
                    ))
        model.add(LSTM(32,
                    return_sequences=False,
                    dropout = dropout,
                    recurrent_dropout=recurrent_dropout,
                    stateful = False))
        model.add(Dense(n_outputs, activation=activation))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        model.summary()
    
        return model
    
    def trainnet(model, x_train, y_train, x_val, y_val):
        
        print('\n\nTraining regressor...')
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint('best_model_lstm.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train,validation_data = (x_val,y_val), epochs=100, batch_size=64, callbacks=[es, mc])
                
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) 
        plt.title('Model loss') 
        plt.ylabel('Loss') 
        plt.xlabel('Epoch') 
        plt.legend(['Train', 'Validation'], loc='upper left') 
        plt.show()
        
        print('Regressor trained')
        
        return model
    
    def validatenet(model, x_val, y_val):
        '''Used during the grid search'''
        # Make predictions on the test set using the best model
        y_pred = model.predict(x_val)
        y_pred = y_pred.reshape(-1)
                
        mse = mean_squared_error(y_val, y_pred)
        
        rmse = root_mean_squared_error(y_val, y_pred)
        
        mae = mean_absolute_error(y_val, y_pred)
        
        r_squared = R_squared(y_val, y_pred)
        
        cc = correlation_coefficient(y_val, y_pred)
        
        fit = FIT(y_val, y_pred)
        
        mard = MARD(y_val, y_pred)
        
        return [mse, rmse, mae, r_squared, cc, fit, mard]
    
    

    def testnet(patient, PH, activation, model, scaler, x_test, y_test, actual_target_test):
        # Make predictions on the test set using the best model
        y_pred = model.predict(x_test)
        y_pred = y_pred.reshape(-1)
        
        # Inverse transform the scaled predictions
        # predictions = scaler.inverse_transform(y_pred)
        x_test_2d_slice = x_test[:,0,:]
        x_test_2d_slice[:,0] = y_pred
        predictions = scaler.inverse_transform(x_test_2d_slice)[:, 0]
        actual_values = np.array(actual_target_test)
        
        # Evaluate the best model
        mse = mean_squared_error(actual_values, predictions)
        print(f'Mean Squared Error (Best Model): {mse}')
        
        rmse = root_mean_squared_error(actual_values, predictions)
        print(f'Root Mean Squared Error (Best Model): {rmse}')
        
        mae = mean_absolute_error(actual_values, predictions)
        print(f'Mean Absolute Error (Best Model): {mae}')
        
        r_squared = R_squared(actual_values, predictions)
        print(f'R Squared (Best Model): {r_squared}')
        
        cc = correlation_coefficient(actual_values, predictions)
        print(f'Correlation Coefficient (Best Model): {cc}')
        
        fit = FIT(actual_values, predictions)
        print(f'FIT (Best Model): {fit}')
        
        mard = MARD(actual_values, predictions)
        print(f'MARD (Best Model): {mard}')
        
        # Visualize predictions vs. actual values
        plt.figure(figsize=(12, 6))
        points = np.arange(0, len(predictions), 1)
        plt.plot(points, actual_values, label='Actual', marker='o')
        plt.plot(points, predictions, label='Predicted', marker='o')
        plt.title(f'Blood Glucose Prediction vs. Actual (LSTM - Best Model) {patient} PH {PH}')
        plt.xlabel('Timestamp')
        plt.ylabel('Blood Glucose Level')
        plt.legend()
        plt.savefig(f'Glucose Prediction/{patient}_PH{PH}.png')
        
        # Write metrics to a text file
        output_file_path = f'metric results/metrics_{patient}_PH{PH}_{activation}.txt'
        with open(output_file_path, 'w') as file:
            file.write(f'Mean Squared Error (Best Model): {mse}\n')
            file.write(f'Root Mean Squared Error (Best Model): {rmse}\n')
            file.write(f'Mean Absolute Error (Best Model): {mae}\n')
            file.write(f'R Squared (Best Model): {r_squared}\n')
            file.write(f'Correlation Coefficient (Best Model): {cc}\n')
            file.write(f'FIT (Best Model): {fit}\n')
            file.write(f'MARD (Best Model): {mard}\n')
        
        print(f"Metrics written to {output_file_path}")
        
        
        
