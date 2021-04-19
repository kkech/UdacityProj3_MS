- [Capstone Project - Azure Machine Learning Engineer Nanodegree - Kechagias Konstantinos](#capstone-project---azure-machine-learning-engineer-nanodegree---kechagias-konstantinos)
  - [Dataset](#dataset)
    - [Overview](#overview)
    - [Task](#task)
    - [Access](#access)
  - [Automated ML](#automated-ml)
    - [Results](#results)
      - [AutoML Run Details widget](#automl-run-details-widget)
      - [AutoML Best Model Run](#automl-best-model-run)
      - [AutoML Best Model Run Properties](#automl-best-model-run-properties)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Results](#results-1)
      - [HyperDrive Run Details widget](#hyperdrive-run-details-widget)
      - [HyperDrive Best Model Run](#hyperdrive-best-model-run)
      - [HyperDrive Best Model Run Properties](#hyperdrive-best-model-run-properties)
  - [Model Deployment](#model-deployment)
      - [Service of HyperDrive model with "Active" deployment state](#service-of-hyperdrive-model-with-active-deployment-state)
  - [Future improvements](#future-improvements)

# Capstone Project - Azure Machine Learning Engineer Nanodegree - Kechagias Konstantinos


In the capstone project I will use the knowledge that I obtained during **Machine Learning Engineer with Microsoft Azure Nanodegree Program** to solve the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)

Some passengers of the titanic were more likely to survive than others. The dataset from Kaggle give information for 871 passengers. A a column indicates if they have survived or not. My target is to make a model that will predict which passengers survived the Titanic.

Here we do this in two different ways:
1. Using AutoML.
2. Using a custom model and tuning its hyperparameters with HyperDrive.

Then, I will compare the performance of both the models and deploy the best performing model.
The deployment is done using the Azure Python SDK, and creates an endpoint that can be accessed through a REST API. This step allows any new data to be evaluated by the model through the service easily.

## Dataset

### Overview

The dataset chosen for this project is [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic). 

Some passengers of the titanic were more likely to survive than others. The dataset from Kaggle give information about 871 passengers including a column that states if they have survived or not. My target is to make a model that will predict which passengers survived the Titanic.

I will use only the "training" data because this dataset have the data with the "Survived" label, which is necessary for the Supervised Learning algorithms that are used in the capstone project.

Find below the data dictionary:

Variable | Definition | Key
------------ | ------------- | -------------
Survived | Survival | integer
Nclass | Ticket class | integer
Name | Name of the passenger | string
Sex | Passenger Sex | string
Age	| Age | integer
Pibsp | # of siblings / spouses aboard the Titanic	| integer
Parch | # of parents / children aboard the Titanic	| integer
Ticket | Ticket number	| string
Fare | Passenger fare | float
Cabin | Cabin number |  string
Embarked | Port of Embarkation | string

The data has been uploaded to this Git repository [this repository](https://raw.githubusercontent.com/kkech/UdacityProj3_MS/master/train.csv).

### Task
In this project, I aim to create a model, **Accuracy** metric, to classify if a passenger survives or not the Titanic.
I examine two approaches:

1) **Using AutoML**:
In this approach, I provided the dataset to AutoML and it automatically did the featurization, tries different algorithms, and test the performance of different models. 

2) **Using HyperDrive**: 
I tested a single algorithm and I created different models by providing different hyperparameters. The chosen algorithm is Logistic Regression using the framework SKLearn. Hyperparameter selection mad using Hyperdrive.

In both cases, best performing model created during runs, had saved and deployed, and the parameters can be checked both in the Azure ML portal and run logs.

The features that I used in this experiment are the ones described in the data dictionary above. However, in the case of the HyperDrive, we manually remove the columns "Name", "Ticket", and "Cabin", "Sex", "Embarked" which are not supported by the Logistic Regression classifier.

### Access

The data has been uploaded to this Git repository [this repository](https://raw.githubusercontent.com/kkech/UdacityProj3_MS/master/train.csv).
To access it in Azure notebooks, we need to download it from an external link into the Azure workspace.

For that, we can use the `Dataset` class, which allows importing tabular data from files on the web.

## Automated ML
For the AutoML run, I created a compute cluster to run the experiment. 

The constructor of `AutoMLConfig` class takes the following parameters:
* `task`: type of ML problem to solve, set as `classification`;
* `compute_target`: cluster where the experiment jobs will run;
* `experiment_timeout_minutes`: 20;
* `training_data`: the dataset loaded; 
* `label_column_name`: The column that should be predicted, which is the "Survived" one; 
* `enable_early_stopping`: makes it possible for the AutoML to stop jobs that are not performing well after a minimum number of iterations; 
* `path`: the full path to the Azure Machine Learning project folder; 
* `featurization`: indicator that featurization step should be done automatically;
* `debug_log`: The log file to write debug information to; 
* `automl_settings`: other settings passed as a dictionary. 
    * `max_concurrent_iterations`: Represents the maximum number of iterations that would be executed in parallel. Set to 9;
    * `primary_metric`: The metric that Automated Machine Learning will optimize for model selection. We chose to optimize for `Accuracy`.

Because AutoML is an automated process that might take a long time, it is a good idea to enable the early stopping. This can help in the cost minimazation. AutoML with enable early stopping option is able to kill jobs that are not performing well, leading to better resource usage

### Results
Among many experiments maded by the AutoML, the best model had an accuracy of **82,38%**.

Voting Ensemble uses multiple models as inner estimators and each one has its unique hyperparameters.


#### AutoML Run Details widget Pending
![automl_run_details_widget](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLRunDetails.png)

#### AutoML Run Details widget Completed
![automl_run_details_widget](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLRunDetailsCompl.png)

#### AutoML Best Model Run
![automl_run_web_gui](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLBestModel.png)

## Model Deployment
The model created by the AutoML deployed in an endpoint.

The expected input type is a JSON with the following format:
```json
"data":
        [
          {
            "PassengerId": integer,
            "Pclass": integer,
            "Age": float,
            "Sex": string,
            "SibSp": integer,
            "Parch": integer, 
            "Fare": float,
            "Embarked": string
          }
        ]
```

#### Service of AutoML deployed model
![automl_deployed_model](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLDeployedService.png)

#### Request body of AutoML endpoint call
![automl_request_body]https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLRequestData.png)


#### Service of AutoML model with "Active" deployment state, scoring URI, and swagger URI. Also, a response from the server is included.
![automl_response](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLResponse_AND_ServiceStatus.png)

#### Service of AutoML model proof of deletion
![automl_service_active](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLProofOfDeleteion_AND_Logging.png)

## Standout
My deployed web app had enabled logging. I used application insights for logging to Monitor and collect data from ML web service endpoint. Logging was enabled programmatically, and the code can be found in the jupyter.

#### Application Insights Logging
![automl_run_logging](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/autoMLProofOfDeleteion_AND_Logging.png)

## Hyperparameter Tuning
I used Hyperpameter Tuning Tool with a Logistic Regression model from the SKLearn framework in order to classify if a passenger would survive or not in the Titanic.
Logistic regression assumes a linear relationship between input and output.
I selected Logistic regression because this will allow me to experiment quickly in the Azure ML environment.

Hyperdrive is used to sample different values for two algorithm hyperparameters:
* `C`: Inverse of regularization strength
* `max_iter`: Maximum number of iterations taken for the solvers to converge

I sample the values using Random Sampling, where hyperparameter values are randomly selected from the defined search space. `C` is chosen randomly in uniformly distributed between **0.001** and **1.0**. `Max_iter` sampled from one of the three values: **1000, 10000, and 100000**.

### Results
 HyperDrive accuracy was **74,43%** which is not so good as the AutoML run.
 
The parameters used by this classifier are the following:
* C = 0.9758520032406058
* Max iterations = 100000

#### HyperDrive Run Parameters to be tuned
![hyperdrive_run_params](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/hyperParamToTune.png)

#### HyperDrive Run Details widget pending
![hyperdrive_run_details_pending](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/hyperRunDetailsPend.png)

#### HyperDrive Run Details widget completed
![hyperdrive_run_details_completed](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/hyperRunDetailsCompl.png)

#### HyperDrive Best Model Run
![hyperdrive_best_run_graph](https://github.com/kkech/UdacityProj3_MS/blob/master/starter_file/screenshots/hyperRunBestModel.png)


## Screen Recording

#### ScreenCast Youtube
I recorded my screen in full screen mode at 1080p and 16:9 aspect ratio. I used OBS for the recording.

![screencast]()


## Future improvements

There are several ways in order to imporve our AutoML and HyperDrive runs.

Firstly, in both runs we could change the  performance metric from `Accuracy` to `AUC_weighted` for example, which could produce better results.

An improvement for AutoML run, is to choose the best 3-5 algorithms and create another AutoML run with only this algorithms. I could also have a look at the data that has been wrongly classified by the best model and try to identify a pattern that could lead to transformations on them. That can be done by creating a pipeline with a first step to transform the data and a second one to execute the AutoML.

An improvement for the HyperDrive run is to test different classifier algorithms in our training script.Also, I could test another algorithm like Random Forests and Decision Trees. For every of those algorithms a different set of hyperparameters can be choose using either Random Sampling or other sampling methods. Deep Learning algorithms could also be applied to solve this problem, and it will be interesting to look into thme.