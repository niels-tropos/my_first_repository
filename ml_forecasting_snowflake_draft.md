# How use Python for machine learning-based forecasting in dbt - Snowflake ecosystem
This blog will discuss the potential of using Python in a native SQL runner such as dt/Snowflake, to enable machine learning in your data projects. This post will go over reasons as to why you would want to run Python in dbt/Snowflake, how that would work and a code example from one of our own projects. More specifically, we recently implemented Facebooks Prophet, an advanced SARIMA model, right into dbt Cloud to forecast client demand in different municipalities based on historic trends.

- [Why Python in dbt?](#why-python-in-dbt)
- [Python models](#python-models)
- [Machine learning models in dbt](#machine-learning-models-in-dbt)
- [Ref case: Time series forecasting to predict client demand](#ref-case-time-series-forecasting-to-predict-client-demand)
- [Final Thoughts](#Final-thoughts)

# Why Python in dbt?
Initially, this may seem strange. Transforming data is typically performed in native SQL runners. Eventhough it is possible with Python, SQL is known to be much more performant when it comes to quickly querying data. However, when it comes to machine learning, Python's rich open source library of pre-build packages allows you to easily implement advanced machine learning techniques right into your projects. While SQL beats Python in terms of raw data querying performance, Python beautifully complements this strength by enabling the implementation of advanced machine learning techniques on that same data. As of recently, both dbt and Snowflake have enabled writing and execution Python code directly into their environments, enabling engineers to effectively use A.I. right in their Snowlake data warehouse. 

enable a mini machine learning pipeline, end to end data processing

# Python models
In dbt, a python model functions exactly as any other SQL model would. It can reference one or more upstream .sql models and can be referenced by downstream models using dbt's built-in ref funtion. Similar to the .sql models, Python models are created by adding the .py suffix and have to reside in dbts models folder. While a typical .sql model would look something like this:
```sql
SELECT *
FROM {{ ref('ml_pre_clientdemand') }}
```
where `ml_pre_clientdemand` is a regular upstream SQL model, a python model has a slightly more complex base structure:
```python
def model(dbt, session):
    dbt.config(materialized = "table", packages = ["pandas"])
    referenced_table = dbt.ref("ml_pre_clientdemand")
    df = referenced_table.to_pandas()
    
    #Python magic here!
    
    return df
```
A few things to note here: 
1. The model parameters `dbt` and `session` are required and not to be changed
2. A dbt config block is used to configure the model as well denote any third party packages like numpy you might want to use
3. The `ORGADMIN` of target Snowflake account must enable the use of third party Python packages --> [Using Third-Party packages in Snowflake](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages.html#using-third-party-packages-from-anaconda)
4. The model has to return a single dataframe, which will be materialized in your (Snowflake) data warehouse
5. Once a SQL model is referenced (`referenced_table`) and converted into a dataframe, all Python is fair game

In the dbt lineage graph, a Python model is indistinguisable from the regular SQL models:
![image](https://user-images.githubusercontent.com/101560764/212186189-c5e7aab7-586e-4b64-8cee-b586118bc2e9.png)

# Snowpark API
Alternatively, the Snowpark API can be used to access and transform tables in Snowflake from a (local) Python IDE. It also enables you to execute Python using Snowflake's compute power. As an examle, we have used the Snowpark API to deploy a Python UDF from a local notebook to our Snowflake account that converts XML formatted cells into the preferred JSON format. Besides UDFs, it is also possible to create Snowflake stored procedures using the Python languag. In fact, this is how dbt runs the Python models in the background. Snowflake converts the Python models into a temporary stored procedure, which is called and executed once before being dropped.

--to be continued--

# Machine learning models
Now that we have a basic understanding of how Python models work in dbt, it is time to take it up a notch. Python's open source library of packages make more advanced matters such as machine learning and A.I. accessible to everyone. Popular packages like `Facebook's Prophet`, `Amazon's deepAR`, `Scikit-learn`, `Scipy`, `Pandas` and even `Pytorch` and `Tensorflow` make it easier than ever for data engineers to implement machine learning in their data projects to solve all kinds of problems. Typical usecases may be:
1. Data clustering to group similar clients 
2. Classification to predict cancer in patients based on a collection of biomarkers
3. Anomaly detection to detect fraud or machine defects
4. Time series forecasting to predict future client demand based on historic data
5. ... 

Since both dbt and Snowflake now allow the full usage of Python, we can set up an <ins>end-to-end</ins> machine learning pipeline right in dbt and Snowflake. The next section will go into more detail about how we use dbt and Snowflake to set up an <ins>end-to-end</ins> machine learning-based forecasting system to predict future client demand for each municipality in Flanders, Belgium.

# Ref case: Time series forecasting to predict client demand
A client active in home care wanted to predict future client demand for each municipality in Flanders in order to steer and match employee availability in order to avoid having employees without clients and having clients without a care taker. By predicting client demand in each municipality, they would be able to take employees who have too little clients from one municipality and deploy them to another where client demand exceeds employee availability. To aid in their migration to the Cloud, the following basic structure was set up:
<img width="749" alt="image" src="https://user-images.githubusercontent.com/101560764/212202267-9f269830-6c09-4710-98f4-901642b71e5d.png">

Using dbt Cloud, source data containg raw client and employee information is ingested into Snowflake. Next, the raw data is modelled using a collection of regular SQL models into a familiar star-schema or party-event model. From the modelled data, flat tables are derived in a preprocessing step and staged to be used in a forecasting algorithm. Here, historic client demand per municipilaty is chronologically ordered and collected in one big table. Once the data has the right shape and format, a Python model containing the Prophet forecasting algorithm is trained on each individual municipality and predicts client demand for the next few months. Here is a code example of the Python model in dbt Cloud:
```python
import pandas as pd
from prophet import Prophet
import numpy as np
from datetime import datetime


def train_predict_prophet(df, periods):
    df_prophet_input = df[['ds', 'y']]
    model = Prophet()
    model.fit(df_prophet_input)
    future_df = model.make_future_dataframe(
        periods=periods, 
        include_history=False)
    forecast = model.predict(future_df)
    return forecast

def min_max_scaling(column):
    if column.min() == column.max():
        return [1] * len(column)
    return (column - column.min())/(column.max() -  column.min())


def model(dbt, session):
    dbt.config(materialized = "table", packages = ["pandas", "numpy", "prophet"])

    my_sql_model_df = dbt.ref("ml_pre_clientdemand")
    df_main = my_sql_model_df.to_pandas() #CONVERT TO DATAFRAME DATATYPE
    df_main['DATE'] = pd.to_datetime(df_main['DATE'], format='%Y-%m-%d') #CONVERT TO CORRECT DATEFORMAT
    df_main = df_main.sort_values(by=['MUNICIPALITY', 'DATUM'])
    df_main = df_main.rename(columns={"DATUM": "ds", "KLANT_VRAAG": "y"}) #RENAME DATUM AND VAL COLUMNS TO DS AND Y FOR PROPHET

    unique_regional_cities = df_main.MUNICIPALITY.unique()
    unieke_regionale_steden= df_main.REGIONALE_STAD.unique()
    union = pd.DataFrame()

    for regionale_stad in unique_regional_cities:
        for gemeente in unique_municipalities:

            df_municipality = df_main.loc[(df_main['REGIONAL_CITY'] == regionale_stad) & (df_main['MUNICIPALITY'] == municipality)]
            if municipality.shape[0] == 0:
                continue

            #SCALAR FOR DENORMALIZATION AND EXTRACT CURRENT REGION
            unique_regions = df_municipality.REGIO.unique()
            current_region = unique_regions[0]

            #NORMALIZE DATASET TO CONSIST OF VALUES 0-1
            df_municipality_history_deep = df_municipality.loc[(df_municipality['ds'] < datetime.strptime("2020-01-01", '%Y-%m-%d'))]
            df_municipality_history = df_municipality.loc[(df_municipality['ds'] >= datetime.strptime("2020-01-01", '%Y-%m-%d')) &       (df_gemeente['ds'] < datetime.strptime("2022-01-01", '%Y-%m-%d'))]
            df_municipality_current = df_municipality.loc[(df_municipality['ds'] >= datetime.strptime("2022-01-01", '%Y-%m-%d'))]

            scalar = df_municipality_current['y'].max() - df_municipality_current['y'].min() 
            term = df_municipality_current['y'].min() #descaling occurs by: scalar * val + ter
            if scalar < 0.001: # aka scalar is zero
                scalar = df_municipality_current['y'].max()
                term = 0

            df_municipality_history_deep['y'] = min_max_scaling(df_municipality_history_deep['y'])
            df_municipality_history['y'] = min_max_scaling(df_municipality_history['y'])
            df_municipality_current['y'] = min_max_scaling(df_municipality_current['y'])
            df_municipality = pd.concat([df_municipality_history_deep, df_municipality_history, df_municipality_current])

            #TRAIN PROPHET AND RETURN FORECAST
            forecast = train_predict_prophet(df_municipality, 160)

            #ADD CHARACTERIZING COLUMNS TO FORECAST
            forecast['REGION'] = [current_region] * len(forecast)
            forecast['REGIONAL_CITY'] = [regional_city] * len(forecast)
            forecast['MUNICIPALTY'] = [municipality] * len(forecast)
            forecast['ISFORECAST'] = [1] * len(forecast)
            forecast['SCALAR'] = [scalar] * len(forecast)
            forecast['TERM'] = [term] * len(forecast)

            #UNION HISTORIC DATA AND FORECAST
            union = pd.concat([union, forecast])

    union = union.reset_index(drop=True)
    union['ds'] = union['ds'].dt.date
    union = union.rename(columns={"ds": "DS", "y": "Y", "yhat":"YHAT", "yhat_lower":"YHAT_LOWER", "yhat_upper":"YHAT_UPPER"})
    
    return union
```
A few things to note here: 
- As can be seen above, while `def model(dbt, session)` function is required, there is no limit on the number of self-defined functions that you can use. 
- Additionally, while as much of the preprocessing as possible should be done in an upstream SQL model for performance purposes, some light preprocessing and postprocessing can be done if the situation calls for it. Examples are casting dateformats to pandas dateformat, renaming columns as Prophet demands the value column to be called `y` and the date column to be called `ds` or collecting the forecast results in a manner easily readible. 
- Lastly, the packages defined in the dbt model's config block and those imported at the top of the file _the old fashioned way_ have the same functionality. However, by importing packages at the top, they allow you to set abreviations for certain names (like pandas --> pd).

Once the output of the forecast has been collected, a postprocessing step is performed in a final SQL model before the result table is send to the visualisation tool:
- <img width="707" alt="image" src="https://user-images.githubusercontent.com/101560764/212207236-cc10b7e8-c384-4dd6-807a-6178d7ba9bea.png">

Performance metrics show accurate predictions for up to a year after the last observation, implying Prophet correctly captures historic trends to predict future client demand. Although Prophet is still a relatively simple forecasting technique, it opens the door to implement much more advanced techniques should the problem demand it. Amazon's deepAR or other recurrent and LSTM neural networks have proven to be more accurate and more capable of learning complex patterns. However, training neural netowrks takes significantly longer. For this reason, performing model training and prediction in a single Python model (as shown above) is discouraged. When working with neural networks, it is much more efficient to train the model in a one Python model and saving the weights in a Pickle file to an internal stage in Snowflake. Next, to use the trained model, the weights can be loaded back in into a different Python model, separation training from prediction.

--provide code example using DeepAR--

# Final thoughts

