# End To End DL Pipeline

## Description

This is an end to end deep learning pipeline for a classification problem. It includes data ingestion, data transformation, data split, model build, model train, and model test. The pipeline supports multiple deep learning model architectures including Convolutional Neural Networks (CNN), Multi-Layer Perceptrons (MLP), and Long Short-Term Memory (LSTM) networks. The model architecture is defined in a JSON file that can be exported and reused across different projects. The model hyperparameters can be configured through the parameters file.

The brief description of various pipeline stages is as below:

## Pipeline Stages 

### Data Ingestion

- Ingest data from source   
- Save data to local

### Data Transformation

- Transform data, like removing null values, data normalization, etc.
- Save data to local

### Data Split

- Split data into train and test
- split data based on test size in the params.yaml file
- Save data to local

### Model Build  

- Build model
- Build models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

### Model Train

- Train model
- Train models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

### Model Test   

- Test model
- Test models like MLP, LSTM, CNN, based on the model type and parameters in the params.yaml file
- Save model to local

## Model Management
The project is integrated with MLflow for model management and experiment tracking. MLflow helps track:

- Model versions and artifacts
- Model parameters and hyperparameters 
- Training metrics and evaluation results
- Model dependencies and environment
- Model deployment status

MLflow provides a centralized repository to:

- Compare different model versions and their performance
- Track experiment history and reproduce results
- Manage model lifecycle from training to deployment
- Share models and results across team members
- Deploy models to production environments

The MLflow tracking server can be accessed at the URI specified in params.yaml. The web interface allows visualization of experiments, metrics and artifacts.


## Pipeline Management

The pipeline stages are managed using Data Version Control (DVC). DVC helps:

- Track data and model dependencies between pipeline stages
- Cache intermediate outputs to avoid redundant computation
- Reproduce the entire pipeline or specific stages
- Version control large data files and model artifacts
- Visualize the pipeline DAG and stage dependencies
- Parallelize independent stages for faster execution

The pipeline stages and their dependencies are defined in dvc.yaml. Running `dvc repro` executes the pipeline in the correct order based on the DAG. DVC ensures reproducibility by tracking the exact versions of data and code used in each stage.


### Example pipeline DAG

```
                              +----------------+    
                              | data_ingestion |    
                              +----------------+    
                                        *           
                                        *           
                                        *           
                            +---------------------+ 
                            | data_transformation | 
                            +---------------------+ 
                                        *           
                                        *           
                                        *           
+---------------+               +------------+      
| model_builder |               | data_split |      
+---------------+          *****+------------+      
        *             *****             *           
        *        *****                  *           
        *     ***                       *           
+---------------+                      **           
| model_trainer |                    **             
+---------------+                 ***               
               ***             ***                  
                  **         **                     
                    **     **                       
                 +------------+                     
                 | model_test |                     
                 +------------+                     
```