
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn 
import scipy 

# Thanks to Danny for sharing this function
def intitial_eda_checks(df):
    '''
    Thanks to Danny for sharing this function!
    take a dataframe
    check if there is duplicates
    check if there is nulls
    '''
    if len(df[df.duplicated(keep=False)]) > 0:
        print(f'Number of duplicates is {df[df.duplicated(keep=False)]}')
        df.drop_duplicates(keep='first', inplace=True)
        print('Warming! df has been mutated!')
    else:
        print('No duplicates found.')

    if df.isnull().sum().sum() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False) 
        total = mask_total[mask_total > 0]

        mask_percent = df.isnull().mean().sort_values(ascending=False) 
        percent = mask_percent[mask_percent > 0] 

        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
        print(f'Total and Percentage of NaN:\n {missing_data}')
    else: 
        print('No NaN found.')

def view_columns_w_many_nans(df, missing_percent=.9):
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent]
    return series.index.to_list()

def drop_columns_w_many_nan(df, missing_percent=.9):
    '''
    Define a funciton that will drop the columns whose missing value bigger than missing_percent
    '''
    # mask_percent = df.isnull().mean()
    # series = mask_percent[mask_percent > missing_percent]
    # list_of_col = series.index.to_list()
    list_of_cols = view_columns_w_many_nans(df, missing_percent=missing_percent)
    df.drop(columns=list_of_cols)
    print(list_of_cols)
    return df

# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
# Reference: https://seaborn.pydata.org/tutorial/axis_grids.html
def histograms_numeric_columns(df, numerical_columns):
    '''
    take df, numerical columns as list
    return group histagrams
    '''
    f = pd.melt(df, value_vars=numerical_columns) 
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g


# adapted from House Prices EDA: https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
def boxplots_categorical_columns(df, categorical_columns, dependant_variable):
    '''
    take df, a list of categorical columns, a dependant variable as str
    return group boxplots of correlations between categorical varibles and dependant variable
    '''
    def boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x=plt.xticks(rotation=90)

    f = pd.melt(df, id_vars=[dependant_variable], value_vars=categorical_columns)
    g = sns.FacetGrid(f, col='variable',  col_wrap=2, sharex=False, sharey=False, height=10)
    g = g.map(boxplot, 'value', dependant_variable)
    return g


def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    plt.figure(figsize=(12, 10))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g



def high_corr_w_dependent_variable(df, dependent_variable, corr_value):
    '''
    Get a dataframe of independant varibles that are highly (e.g. abs(corr) > 0.4) with dependent varible
    '''
    temp_df = df.corr()[[dependent_variable]].sort_values(by=dependent_variable, ascending=False)
    mask = abs(temp_df[dependent_variable]) > corr_value
    return temp_df.loc[mask]



def high_corr_among_independent_variable(df, dependent_variable, corr_value):
    '''
    Check correlation among independant varibles (not with SalePrice)
    To see which two features have strong corr with each ohter 
    '''
    df_corr = df.drop(columns=[dependent_variable]).corr()
    corr_dict = df_corr.to_dict()
    temp_dict = {key_1: {key_2 : value 
                         for key_2, value in imbeded_dictionary.items() 
                         if abs(value) < 1 and abs(value) > corr_value}
                for key_1, imbeded_dictionary in corr_dict.items()}
    return {k:v for k, v in temp_dict.items() if v}


def get_categorical_columns(df):
    return [f for f in df.columns if df.dtypes[f] == 'object']

def get_numerical_columns(df):
    return [f for f in df.columns if df.dtypes[f] != 'object']

def dummify_categorical_columns(df):
    '''
    Dummify all categorical columns
    '''
    categorical_columns = get_categorical_columns(df)
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def conform_columns(df_reference, df):
    '''
    Drop columns in df that are not in df_reference
    '''
    to_drop = [c for c in df.columns if c not in df_reference.columns]
    return df.drop(to_drop, axis=1)