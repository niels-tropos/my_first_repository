# How use Python for machine learning-based forecasting in dbt Cloud/Snowflake
This blog will discuss the potential of using Python in a native SQL runner such as dt/Snowflake, to enable machine learning in your data projects. This post will go over reasons as to why you would want to run Python in dbt/Snowflake, how that would work and a code example from one of our own projects. More specifically, we recently implemented Facebooks Prophet, an advanced SARIMA model, right into dbt Cloud to forecast client demand in different municipalities based on historic trends.

- [Why Python in dbt?](#why-python-in-dbt)
- [Python models](#python-models)
- [Machine learning models in dbt](#machine-learning-models-in-dbt)
- [Ref case: Time series forecasting to predict client demand](#ref-case:-time-series-forecasting-to-predict-client-demand)
- [Final Thoughts](#Final-thoughts)

# Why Python in dbt?
Initially, this may seem strange. Transforming data is typically performed in native SQL runners. Eventhough it is possible with Python, SQL is known to be much more performant when it comes to quickly querying data. However, when it comes to machine learning, Python's rich open source library of pre-build packages allows you to easily implement advanced machine learning techniques right into your projects. While SQL beats Python in terms of raw data querying performance, Python beautifully complements this strength by enabling the implementation of advanced machine learning techniques on that same data. As of recently, both dbt and Snowflake have enabled writing and execution Python code directly into their environments, enabling engineers to effectively use A.I. right in their Snowlake data warehouse. 

# Python models

# Machine learning models

# Ref case: Time series forecasting to predict client demand

# Final thoughts


# UDF_xml2json
**xml2json** is a Snowflake UDF specifally designed to efficiently and easily convert xml to json. It uses the Snowpark API to enable Python code to run in Snowflake, deploying a permanent UDF that converts XML to JSON. Executing the code in the source_code.py file deploys the permanent UDF to the target Snowflake account and makes is accessible to the specified users and roles. Cross account sharing of the function is currently still handled within Snowflake. 

- [Set-up](#set-up)
- [Code usage and application](#code-usage-and-application)
- [Code breakdown](#code-breakdown)

# Set-up
To run the code on your own machine and deploy the UDF yourself, note the following steps:
1. Install Anaconda Navigator on your machine: [Anaconda](https://www.anaconda.com/products/distribution).
2. Create a new environment with Python version 3.8.13, ***not*** the newest 3.9.
3. Open the terminal and activate your new environment: `conda activate <your env name>`.
4. Download the 'requirements.txt' file and run the following command in the terminal to install all necessary packages: `pip install -r requirements.txt`. Alternatively, run ```pip install snowflake-snowpark-python``` and ```conda install xmltodict```.
5. You may close the terminal now.
6. Open the 'source_code.py' file in an IDE of your choosing in your Snowflake environment created in step 2.
7. Enter the connection parameters in the first section of the code file. For this, follow the in-code guidelines.
8. NOTE: Before running the code, the ORGADMIN must have allowed third party packages originating from Anaconda to be used in the target snowflake account. If the option is not enable, the code will not execute. For enable this function, go to [this link](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages.html) and follow the instructions.

# Code usage and application
Executing the 'source_code.py' file will deploy a new **xml2json** permanent UDF to target Snowflake account and will override any previously existing UDFs with the same name. For this, you need to enter the connection paramters in between the brackets: 
```python
account = ""
login_name = ""
password = ""
role = ""
warehouse = ""
database = ""
schema = ""
user_stage = ""
privileged_roles_to_access_udf = ["a, b, ..."]
```
In Snowflake, the function simply converts xml to json. A typical usecase would be:
```sql
SELECT *, xml2json(xml)
FROM <table>
```
Additionally, when creating a new table, make sure you denote the column names of the new table you want to create. Example:
```sql
CREATE OR REPLACE table <new table> (column1, column2, column3, ..., JSON) AS
SELECT *, xml2json(xml) AS JSON
from <old table>
```
***Importantly***, if the input XML format is invalid, the function will output the text "*Error: invalid XML format*" to the row in question. This means that the function will not stop and simply continue without visibly raising an error. To verify the conversion succeeded,  you need to check manually whether there are any rows where the conversion failed by applying the `... WHERE <JSON column> like "%Error: invalid XML format%"` filter on the newly created table. Should this error occur in cases where the XML format is valid, you may report this as a bug to [Niels Palmans](https://github.com/niels-tropos).

# Code breakdown
This section will discuss the code in greater detail, going over each seperate block and explaining its function.

First, the necessary packages are imported, specifically Snowpark to establisch the connection between your local Python code and the Snowflake environment. Fill in the Snowflake connection parameters to exactly tell the system where to deploy the xml2json UDF to. These parameters control what Snowflake account the function will be deployed on, on what stage it is staged and what roles can use it.
```python
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf
import xmltodict, json

#-------------------------------------------Fill in Snowflake Connection Parameters------------------------------------------

account = "***"
login_name = ""***""
password = ""***""
role = "accountadmin"
warehouse = "engineering"
database = "demo"
schema = "openstreetmap"
user_stage = "@~"
privileged_roles_to_access_udf = ["engineer"]

```
Next, we define a function that will later be used to initialize a new session object using the connection parameters set in the previous step. When the function is called, it will also print out the current warehouse, database and schema the newly created session is running on.
```python
def create_session_object(connection_parameters):
    current_session = Session.builder.configs(connection_parameters).create()
    print(current_session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    return current_session
```

Third, we define the actual xml2json function that will convert the xml formatted strings into easier to process json. If the conversion fails due to the detection of an invalid xml format, the function will not stop but rather return "Error: invalid XML format" and continue. This is to avoid the function crashing an entire query when encountering a single cell containing an invalid xml format.
```python
def xml2json(xml: str)-> str:
    try:
        json_string = json.dumps(xmltodict.parse(xml))
        json_string_cleaned = json_string.replace("@", "")
        return json_string_cleaned
    
    except Exception as e:
        return "Error: invalid XML format"
```

Now that we have set the connection parameters, made a function to create a new session and defined the xml2json function, we will use the connection parameters to create a new session? Afterwards, we import the relevant `xmltodict` package into Snowflake to make sure Snowflake has access to this package.
```python
connection_parameters = {
"account": account,
"user": login_name,
"password": password,
"role": role,
"warehouse": warehouse,
"database": database,
"schema": schema
}

current_session = create_session_object(connection_parameters)
current_session.add_packages("xmltodict")
```

Next, we register the xml2json function in Snowflake using the register method. This converts the above Python code into an actual usable permanent UDF in Snowflake. The replace parameter is set to `True` as to make sure that rerunning the code actively updates the UDF with the latest version. 
```python
current_session.udf.register(xml2json
                             , name ="XML2JSON"
                             , is_permanent=True
                             , stage_location=user_stage
                             , replace=True)
```

Lastly, we grant the usage privilage to each role that mentioned in the `privileged_roles_to_access_udf` parameter. Here, we use the Snowpark API to write SQL that is to be executed in Snowflake.
```python
for role in privileged_roles_to_access_udf:
    current_session.sql(f'grant usage on function xml2json(varchar) to {role}').collect()
```

