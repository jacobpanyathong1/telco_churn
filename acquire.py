# Acquire File 
import os       # import os filepaths
import env      #importing get_connection function
import pandas as pd     #import Pandas library as pd

def get_telco():    #Creating get_telco function
    '''
    This function is creating the  the filename telco.csv and returning the sql query for
    telco database that results in our DataFrame.
    '''
    filename = 'telco.csv'      #Get Data

    if os.path.isfile(filename):    #searching os path for 'filename'
        return pd.read_csv(filename)    #filename of csv for telco churn dataframe
    else:       #Creating False statement for if-else statment
        df = pd.read_sql(#Creating SQL Query from telco database.
        '''    
        SELECT *
        FROM customers AS a
        JOIN contract_types as b
        USING (contract_type_id)
        JOIN internet_service_types as c
        ON a.internet_service_type_id = c.internet_service_type_id
        lEFT JOIN payment_types as d
        ON a.payment_type_id = d.payment_type_id;
        ''', env.get_connection('telco_churn'))    #Retrieving SQL Query into DataFrame
        df.to_csv(filename)
        return df       #Closing the statement and returning the Telco Dataframe
        