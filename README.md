# End To End DL Pipeline

## Description

This is an end to end deep learning pipeline for a classification problem. It includes data ingestion, data transformation, data split, model build, model train, and model test. The pipeline supports multiple deep learning model architectures including Convolutional Neural Networks (CNN), Multi-Layer Perceptrons (MLP), and Long Short-Term Memory (LSTM) networks. The model architecture is defined in a JSON file that can be exported and reused across different projects. The model hyperparameters can be configured through the parameters file.

The brief description of various pipeline is as below:

## Data Ingestion

- Ingest data from source   
- Save data to local

## Data Transformation

- Transform data, like removing null values, data normalization, etc.
- Save data to local

## Data Split

- Split data into train and test
- split data based on test size in the params.yaml file
- Save data to local

## Model Build  

- Build model
- Build models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

## Model Train

- Train model
- Train models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

## Model Test   

- Test model
- Test models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

