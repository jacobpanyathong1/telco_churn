#Prepare data
import os       # import os filepaths
import env      #importing get_connection function
import pandas as pd     #import Pandas library as pd
import sklearn.model_selection
from scipy import stats
import acquire
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def telco():
    df = acquire.get_telco()
    return df 

def telco_df():
    '''
    Acquiring and preparing the telco database in one function.
    '''
    df = acquire.get_telco()
    df = prep_telco(df)
    return df


def prep_telco(df):
    '''
    #This function will create the cleaning data tasks of the Telco Churn DataFrame 
    #and return our Telco churn DataFrame without objects and nulls.
    '''
    encoded_vars = pd.get_dummies(
        df[['contract_type', 
        'multiple_lines', 
        'online_security', 
        'online_backup',
        'device_protection',
        'tech_support',
        'streaming_movies',
        'streaming_tv',
        'paperless_billing',
        'internet_service_type', 
        'payment_type']])
    df.drop(columns=[
        'customer_id',
        'internet_service_type_id.1', 
        'payment_type_id.1',
        'contract_type',
        'internet_service_type',	
        'payment_type', 
        'Unnamed: 0'
        ], inplace=True
        ) #Drop duplicate columns
    pd.set_option('display.max_columns', 500) # Display all columns
    df = pd.concat([df, encoded_vars], axis=1)
    df['total_charges'] = (df.total_charges + '0').astype('float') # Change total charges to float
    df['total_charges'].dtype
    df['gender'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['churn'] = df.churn.map({'Yes': 1, 'No': 0})
    df.drop(columns=[
        'multiple_lines',
        'online_security', 
        'online_backup',
        'device_protection',
        'tech_support',	
        'streaming_tv', 
        'streaming_movies',
        'paperless_billing'
        ], inplace=True
        ) #Drop columns
    df.rename(columns={'contract_type_Month-to-month': 'month_to_month_contract', 'contract_type_One year': 'one_year_contract', 'contract_type_Two year': 'two_year_contract', 'internet_service_type_DSL': 'IST_DSL'}, inplace=True)
    return df

def prep_telco1(df): # used to create customer_id column for X_test 
    '''
    This function will create the cleaning data tasks of the Telco Churn DataFrame 
    and return our Telco churn DataFrame without objects and nulls.

    //includes customer_id column
    '''
    encoded_vars = pd.get_dummies(
        df[['contract_type', 
        'multiple_lines', 
        'online_security', 
        'online_backup',
        'device_protection',
        'tech_support',
        'streaming_movies',
        'streaming_tv',
        'paperless_billing',
        'internet_service_type', 
        'payment_type']])
    df.drop(columns=[
        'internet_service_type_id.1', 
        'payment_type_id.1',
        'contract_type',
        'internet_service_type',	
        'payment_type', 
        'Unnamed: 0'
        ], inplace=True
        ) #Drop duplicate columns
    pd.set_option('display.max_columns', 500) # Display all columns
    df = pd.concat([df, encoded_vars], axis=1)
    df['total_charges'] = (df.total_charges + '0').astype('float') # Change total charges to float
    df['total_charges'].dtype
    df['gender'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['churn'] = df.churn.map({'Yes': 1, 'No': 0})
    df.drop(columns=[
        'multiple_lines',
        'online_security', 
        'online_backup',
        'device_protection',
        'tech_support',	
        'streaming_tv', 
        'streaming_movies',
        'paperless_billing'
        ], inplace=True
        ) #Drop columns
    df.rename(columns={'contract_type_Month-to-month': 'month_to_month_contract', 'contract_type_One year': 'one_year_contract', 'contract_type_Two year': 'two_year_contract', 'internet_service_type_DSL': 'IST_DSL'}, inplace=True)
    return df

def point_plot():
    df = telco()
    column_x= df['payment_type']
    sns.pointplot(data=df, x=column_x, y=df.tenure, hue=df.senior_citizen)
    plt.xticks([0, 1, 2, 3], ['Check', 'Echeck', 'Credit', 'Direct deposit'])
    plt.title("Average Tenure of Payment Type By Senior Citizen")
    return

def churn_contract_barplot():
    df = telco()
    sns.barplot(x=df.churn, y=df.tenure, hue=df.contract_type, data=df)
    churn_rate = df.tenure.mean()
    plt.axhline(churn_rate, label='tenure rate')
    plt.title("Average month of tenure for churn on contract type")
    plt.legend()
    plt.show()
    return

def churn_swarmplot():
    df = telco()
    plt.title("Average Monthly Charges for Churn by Internet Service Type")
    sns.set_theme(style="ticks")
    sns.swarmplot(data=df, x="monthly_charges", y="churn", hue="internet_service_type")
    plt.show()
    return

def boxen_plot():
    df = telco()
    column_y = df['payment_type']
    sns.boxenplot(data=df, x=df['tenure'], y=column_y, showfliers=True)
    plt.title("Average tenure of customers by payment type")
    plt.show()
    return 


def train_test_split(df, target):

    train_validate, test =  sklearn.model_selection.train_test_split(df, test_size=.2, random_state = 100, stratify = df[target])
    train, validate = sklearn.model_selection.train_test_split(train_validate, test_size=.3, random_state = 100, stratify= train_validate[target])
    
    return train, test, validate


def create_train_validate_test_samples(train, validate, test):
    X_train = train.drop(columns='churn')
    y_train = train.churn

    X_validate = validate.drop(columns='churn')
    y_validate = validate.churn

    X_test = test.drop(columns='churn')
    y_test = test.churn

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def alpha():
    alpha = 0.05
    return alpha

def null_hyp_variables(variable_1, variable_2):
    null_hypothesis = print(f'there is no significant relationship between {variable_1} and {variable_2} are significantly dependent on each other')
    return null_hypothesis

def alt_hyp_variables(var1, var2):
    alt_hypothesis = print(f'there is a significant relationship between {var1} and {var2}')
    return alt_hypothesis

def observed(pred, target, actual, data):
    crosstab = pd.crosstab(pred[target], actual[data])
    return crosstab

def chi2_test(observed):
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    return chi2, p, degf, expected

def chi_test_findings(p, alpha, null_hyp, alt_hyp):
    if p < alpha:
        print(f'We reject the null hypothesis that, {null_hyp}')
        print(alt_hyp)
    else:
        print(f'We fail to reject the null hypothesis that, {null_hyp}')
        print(null_hyp)

def churn_contract_type():
    df = acquire.get_telco()
    observe= observed(df, 'churn', df, 'contract_type')
    chi2, p, degf, expected = chi2_test(observe)
    df= degf
    df = expected
    t= print(f't = {chi2}')
    p_value =print(f'p = {p}')
    return t, p_value

def churn_internet_service():
    df = acquire.get_telco()
    observe= observed(df, 'churn', df, 'internet_service_type')
    chi2, p, degf, expected = chi2_test(observe)
    df= degf
    df = expected
    t= print(f't = {chi2}')
    p_value =print(f'p = {p}')
    return t, p_value

def baseline_accuracy():
    df = acquire.get_telco()
    train, validate, test =train_test_split(df, 'churn')
    train['baseline'] = 0
    baseline_accuracy = (train.baseline == train.churn).mean()
    v = validate
    t = test
    return baseline_accuracy
    

def rfc_model():
    df = acquire.get_telco()
    df = prep_telco(df)
    train, validate, test = train_test_split(df, 'churn')
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_train_validate_test_samples(train, validate, test)
    rfc = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=5, 
                            random_state=120)
    rfc.fit(X_train, y_train)
    rfc.score(X_test, y_test)
    y_pred =rfc.predict(X_train)
    y_pred_proba = rfc.predict_proba(X_train)
    #create an empty list to append results to
    metrics = []

    for j in range (1, 10):
        for i in range(2, 10):
            rf = RandomForestClassifier(max_depth=i, 
                                    min_samples_leaf=j, 
                                    random_state=123)

            # Fit the model (on train and only train)
            rf = rf.fit(X_train, y_train)

            # We'll evaluate the model's performance on train, first
            in_sample_accuracy = rf.score(X_train, y_train)
    
            out_of_sample_accuracy = rf.score(X_validate, y_validate)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_accuracy": in_sample_accuracy,
                "validate_accuracy": out_of_sample_accuracy
            }
    
            metrics.append(output)
            # create a df from metrics
            df = pd.DataFrame(metrics)

            # compute difference in accuracy between train and validate
            df["difference"] = df.train_accuracy - df.validate_accuracy

            # sort the df by validate_accuracy (descending) and take top 10
            df = df.sort_values(by=['validate_accuracy'], ascending=False).head(10)
            acc = df.iloc[0][2:4]
            train_acc = acc.index[0]
            validate_acc = acc.index[1]
            train_model = acc[0]
            validate_model = acc[1]
            a = print(f'{train_acc}: {train_model}')
            b = print(f'{validate_acc}: {validate_model}')
            return a, b

def dtc_model():
    df = acquire.get_telco()
    df = prep_telco(df)
    train, validate, test = train_test_split(df, 'churn')
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_train_validate_test_samples(train, validate, test)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=100)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_train)
    tree2 = DecisionTreeClassifier(max_depth=4)
    tree2.fit(X_train, y_train)
    
    metrics = []

    for i in range (1, 25):
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)
        tree = tree.fit(X_train, y_train)
        in_sample_accuracy = tree.score(X_train, y_train)
        out_of_sample_accuracy = tree.score(X_validate, y_validate)
    
        output = {"max_depth": i, "train_accuracy": in_sample_accuracy, "validate_accuracy": out_of_sample_accuracy}
    
        metrics.append(output)
    
        df = pd.DataFrame(metrics)
        df['difference'] = df.train_accuracy - df.validate_accuracy
        df
        acc = df.iloc[0][1:3]
        train_acc = acc.index[0]
        validate_acc = acc.index[1]
        train_model = acc[0]
        validate_model = acc[1]
        a = print(f'{train_acc}: {train_model}')
        b = print(f'{validate_acc}: {validate_model}')
        return a, b

def log_model():
    df = acquire.get_telco()
    df = prep_telco(df)
    train, validate, test = train_test_split(df, 'churn')
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_train_validate_test_samples(train, validate, test)
    logit = LogisticRegression(random_state=123, solver='liblinear')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_train)
    y_pred_proba = logit.predict_proba(X_train)
    logit2 = LogisticRegression( random_state=123, solver='liblinear')
    logit2.fit(X_train, y_train)
    y_pred2 = logit2.predict(X_train)
    y_pred_proba2 = logit2.predict_proba(X_train)
    y_pred1 = logit.predict(X_validate)
    y_pred2 = logit2.predict(X_validate)
    y_pred = logit.predict(X_test)
    y_pred_proba = logit.predict_proba(X_test)
    y_pred_proba = np.array([i[1] for i in y_pred_proba])
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(logit2.score(X_train, y_train)))