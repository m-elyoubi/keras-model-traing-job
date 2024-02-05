# # train.py

# import os
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# import boto3
# from tensorflow.keras.models import load_model


# # Load data from S3
# def load_data():
#     # Implement your code to load data from S3
#     # Example: load data from 's3://your-s3-bucket/path/to/data'
#     pass

# # Define your Keras model
# def create_model():
#     model = Sequential()
#     # Add your model architecture
#     return model

# # Train and save the model
# def train_and_save_model():
#     # Load data
#     X_train, y_train = load_data()

#     # Create and compile Keras model
#     model = create_model()
#     model.compile(optimizer=Adam(), loss='mean_squared_error')

#     # Train the model
#     model.fit(X_train, y_train, epochs=10)

#     # Save the trained model to S3
#     model.save("s3://your-s3-bucket/path/to/save/model/model.h5")

# if __name__ == "__main__":
#     train_and_save_model()


# -------------------------------
# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout
import joblib
import os
import sagemaker
import boto3
from io import StringIO
import argparse
import json

s3 = boto3.client('s3')

# Load data from S3
def load_data(bucket_name,file_key):
    # Download the CSV file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    # Convert the CSV content to a Pandas DataFrame
    df = pd.read_csv(StringIO(csv_content))
    return df

def create_lstm_model(lstm_units=150, dropout_rate=0.2, learning_rate=0.001, epochs=3, batch_size=64, X_train=None,
                      y_train=None):
    """Creates and trains an LSTM model."""
    model = keras.Sequential()
    model.add(layers.LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(layers.LSTM(units=lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1, activation='linear'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size, epochs)
    return model


def train_and_predict_stock_price(rates_frame, symbol_name, s3_output_path):
    """Preprocesses data, trains the model, and saves it."""
    try:
        if rates_frame is not None:
            dataset_train, _ = train_test_split(rates_frame, test_size=0.25, shuffle=False)
            training_set = dataset_train.iloc[:, 4:5].values
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            X_train, y_train = [], []

            for i in range(60, len(dataset_train)):
                X_train.append(training_set_scaled[i-60:i, 0])
                y_train.append(training_set_scaled[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Use SageMaker-compatible paths
            model_filename = os.path.join('/opt/ml/model', f'model_{symbol_name}.h5')
            scaler_filename = os.path.join('/opt/ml/model', f'scaler_{symbol_name}.joblib')

            model = create_lstm_model(lstm_units=150, dropout_rate=0.2, learning_rate=0.001, epochs=3, batch_size=64,
                                      X_train=X_train, y_train=y_train)
            
            # Save model to S3
            model.save(model_filename)
            joblib.dump(sc, scaler_filename)

            # Upload to S3
            s3_output_model_path = os.path.join(s3_output_path, f'model_{symbol_name}.h5')
            s3_output_scaler_path = os.path.join(s3_output_path, f'scaler_{symbol_name}.joblib')
            
            sagemaker.s3.S3Uploader.upload(model_filename, s3_output_model_path)
            sagemaker.s3.S3Uploader.upload(scaler_filename, s3_output_scaler_path)

            return s3_output_model_path, s3_output_scaler_path
            
        else:
            # Handle the case when rates_frame is None
            print("Error: rates_frame is None.")
            return None, None
			
    except Exception as e:
        # Handle other exceptions if necessary
        print(f"An error occurred: {str(e)}")
        return None, None


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--symbol_name', type=str, required=True)
    parser.add_argument('--input_file_key', type=str, required=True)
    parser.add_argument('--s3_output_path', type=str, required=True)

    return parser.parse_known_args()


if __name__ == "__main__":

    
    args, unknown = _parse_args()
    print(args)
    
    rates_frame = load_data(args.bucket_name, args.input_file_key)
    train_and_predict_stock_price(rates_frame, args.symbol_name, args.s3_output_path)



    # bucket_name=    'sagemaker-us-east-2-481102897331'
    # symbol_name =   'US30'
    # input_file_key=     f'data-input/{symbol_name}.csv'
    # s3_output_path =     f"s3://{bucket_name}/data-output/"
    # s3_input_path =     f"s3://{bucket_name}/data-input/"
    
    
    # rates_frame = load_data(bucket_name,input_file_key)
    # train_and_predict_stock_price(rates_frame, symbol_name, s3_output_path)
