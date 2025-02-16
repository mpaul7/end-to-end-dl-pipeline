# end-to-end-dl-pipeline

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