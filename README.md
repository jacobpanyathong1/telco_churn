# Project Description

Telco Churn was identified for data science analysis and what KPIs can stakeholders focus on to drive efficencies in business operations. Churn affects business operations as customers leave at a the calculated monthly rate resulting in loss of sales revenue. I am exploring the possible KPI drivers that will influence business decisions to counter customer churn. 

# Project Goal
* Find drivers of churn at telco.
* Construct a ML classification model that accurately predicts customer churn.
* Report that a non-data scientist can read through and understand what steps were taken in my project.

# Initial Hypothesis
My initial hypothesis is that customer churn is dependent on contract type. 

# Project Planning 
* Create acquire file for data

* Create prepare file for functions
    * Create columns for data exploring analysis.
        * 
        
* Drivers for churn 
    * Does contract type affect churn?
    * Do we know if increasing costs affect churn?
    * Does having add-on vs no add-on services affect churn?

* Develop a model that will predict churn.

* Draw conclusions

# Data Dictionary
| Feature | Dictionary |
|:--------|:-----------|
|contract_type_id| The id number of each contract type for each customer in telco churn.
|customer_id| The id number of each customer 
|senior_citizen|	Determines a 1 for is a senior citizen and a 0 for not a senior citizen
|internet_service_type_id|	The id for each internet service product offered.
|payment_type_id|	The id for each payment type for customers.
|monthly_charges|	The amount of monthly charges of each customer.
|total_charges| The total amounnt from each customer
|Tenure| The number of total months a customer churns.
|gender| The gender of each customer M /or F
|partner| The number of each customer with a significant other on record.
|dependents| The number of each customer with children on record.
|phone_service| The number of each customer with a phone line added into their service.
|multiple_services| The category of each customer with more than one phone service and no service at all.
|online_security| The number of each customer with added online security into their service.
|device_protection| The number of each customer with added device protection into their service.
|tech_support| The number of each customer with added tech support into their service.
|streaming_tv| The number of each customer with added streaming tv into their service.
|streaming_movies| The number of each customer with added streaming movies into their service.
|paperless_billing| The number of each customer with added paperless billing into their service.
|churn| The number of each customer that has churned when the data was found.
|contract_type| The type of contract(month to month, one year, two year) each customer has on record. 
