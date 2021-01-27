import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

import statsmodels.api as sm



def intitial_eda_checks(df):
    '''
    Checks duplicates: if any duplicates found, the duplicates will be dropped
    and a warming of dataframe mutation will be issued.
    Checks nulls
    Takes dataframe
    '''
    if len(df[df.duplicated(keep=False)]) > 0:
        print(df[df.duplicated(keep=False)])
        df.drop_duplicates(keep='first', inplace=True)
        print('Warning! Dataframe has been mutated!')
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
    '''
    Checks which columns have over specified percentage of missing
    values 
    Takes dataframe, missing percentage (default=.9)
    Returns columns as a list
    '''
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent]
    columns = series.index.to_list()
    print(columns) 
    return columns



def drop_columns_w_many_nans(df, missing_percent=.9):
    '''
    Drops the columns whose missing value are bigger than the specified missing percentage
    Takes dataframe, missing percentage (default=.9)
    Returns dataframe
    '''
    series = view_columns_w_many_nans(df, missing_percent=missing_percent)
    list_of_cols = series.index.to_list()
    df.drop(columns=list_of_cols)
    print(list_of_cols)
    return df



# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
# Reference: https://seaborn.pydata.org/tutorial/axis_grids.html
def histograms_numeric_columns(df, numerical_columns):
    '''
    Takes dataframe, numerical columns as list
    Returns group histagrams
    '''
    f = pd.melt(df, value_vars=numerical_columns) 
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g


# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
def boxplots_categorical_columns(df, categorical_columns, dependant_variable):
    '''
    Takes dataframe, categorical columns as list, dependant variable as string
    Returns group boxplots of correlations between categorical varibles and dependant variable
    '''
    def boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x=plt.xticks(rotation=90)

    f = pd.melt(df, id_vars=[dependant_variable], value_vars=categorical_columns)
    g = sns.FacetGrid(f, col='variable',  col_wrap=2, sharex=False, sharey=False, height=10)
    g = g.map(boxplot, 'value', dependant_variable)
    return g



def scatter_plots(df, numerical_cols, target_col):
    '''
    Take dataframe, numerical columns as list, target column as string
    Return a group of scatter plots
    '''
    # Calculate the number of rows
    num_rows = (len(numerical_cols) // 3) + 1
    # Generate a 3 x n subplots frame
    fix, ax = plt.subplots(num_rows, 3, sharey='row', figsize=(15,20))

    # Reference: https://stackoverflow.com/a/434328
    # Define a function to iterate through a list and divide them into chunks
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    # Iterate through numerical_cols and generate each subplot
    for y, plot_group in enumerate(chunker((numerical_cols), 3)):
        for x, col in enumerate(plot_group):
            sub_ax = ax[y][x]
            plots = sub_ax.scatter(df[col], df[target_col], s=2)
            plots_titles = sub_ax.set_title(col)
    return (plots, plots_titles)
    


def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    '''
    Takes dataframe, dependant variable as string
    Returns heatmap of independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g



def high_corr_w_dependent_variable(df, dependent_variable, corr_value):
    '''
    Takes dataframe, dependent variable, and value of correlation 
    Returns dataframe of independant varibles that are highly (e.g. abs(corr) > 0.4) with dependent varible
    '''
    temp_df = df.corr()[[dependent_variable]].sort_values(by=dependent_variable, ascending=False)
    mask_1 = abs(temp_df[dependent_variable]) > corr_value
    return temp_df.loc[mask_1]



def high_corr_among_independent_variable(df, dependent_variable, corr_value):
    '''
    Checks correlation among independant varibles, and checks which two features have strong correlation
    Takes dataframe, dependent variable, and value of correlation 
    Returns dictionary 
    '''
    df_corr = df.drop(columns=[dependent_variable]).corr()
    corr_dict = df_corr.to_dict()
    temp_dict = {key_1: {key_2 : value 
                         for key_2, value in imbeded_dictionary.items() 
                         if abs(value) < 1 and abs(value) > corr_value}
                for key_1, imbeded_dictionary in corr_dict.items()}
    return {k:v for k, v in temp_dict.items() if v}



def categorical_to_ordinal_transformer(categories):
    '''
    Returns a function that will map categories to ordinal values based on the
    order of the list of `categories` given. 
    Example: 
    If categories is ['A', 'B', 'C'] then the transformer will map 
    'A' -> 0, 'B' -> 1, 'C' -> 2.
    '''
    return lambda categorical_value: categories.index(categorical_value)



def transform_categorical_to_numercial(df, categorical_numerical_mapping):
    '''
    Transforms categorical columns to numerical columns
    Takes dataframe, dictionary 
    Returns dataframe
    '''
    transformers = {k: categorical_to_ordinal_transformer(v) 
                    for k, v in categorical_numerical_mapping.items()}
    new_df = df.copy()
    for col, transformer in transformers.items():
        new_df[col] = new_df[col].map(transformer).astype('int64')
    return new_df



def dummify_categorical_columns(df):
    '''
    Dummifies all categorical columns
    Takes dataframe
    Returns dataframe
    '''
    categorical_columns = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)



def conform_columns(df_reference, df):
    '''
    Drops columns in dataframe that are not in the reference dataframe
    Takes dataframe as reference, dataframe
    Returns dataframe
    '''
    to_drop = [c for c in df.columns if c not in df_reference.columns]
    return df.drop(to_drop, axis=1)



def vizResids(model_title, X, y, random_state_number=42):
    '''
    Thanks to Mahdi Shadkam-Farrokhi for creating this visualization function!
    Takes model title as string, X(features), y(target)
    Returns 3 error plots 
    '''
    
    # For help with multiple figures: https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/subplots_demo.html

    # HANDLING DATA
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_number)

    # instatiate model
    lr = LinearRegression()
    # fit model
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    resids = y_test - preds
    target_name = y.name.capitalize()

    # HANDLING SUBPLOTS
    fig, axes = plt.subplots(2, 2, figsize=(12,10)) # 2 row x 2 columns
    fig.suptitle(f"{model_title}: $R^2$ test ={lr.score(X_test, y_test):2.2%}", fontsize = 24, y = 1.05)

    ax_1 = axes[0][0]
    ax_2 = axes[0][1]
    ax_3 = axes[1][0]

    subplot_title_size = 18
    subplot_label_size = 14
    
    # 1ST PLOT - y_true vs. y_pred
    ax_1.set_title("True Values ($y$) vs. Predictions ($\hat{y}$)", fontsize = subplot_title_size, pad = 10)
    maxDist = max(max(preds),max(y)) # maxiumum value used to determin x_lim and y_lim
    minDist = min(min(preds),min(y)) # maxiumum value used to determin x_lim and y_lim
    # 45deg line, signifying prediction == true value
    ax_1.plot((minDist,maxDist),(minDist,maxDist), c = "r", alpha = .7);
    
    sns.scatterplot(ax = ax_1, x = y_test, y = preds, alpha = .5)
    ax_1.set_xlabel("True Values ($y$)", fontsize = subplot_label_size, labelpad = 10)
    ax_1.set_ylabel("Predictions ($\hat{y}$)", fontsize = subplot_label_size, labelpad = 10)

    # 2ND PLOT - residuals
    ax_2.set_title("Residuals", fontsize = subplot_title_size)
    sns.scatterplot(ax = ax_2, x = range(len(resids)),y = resids, alpha = .5)
    ax_2.set_ylabel(target_name, fontsize = subplot_label_size)
    ax_2.axhline(0, c = "r", alpha = .7);

    # 3RD PLOT - residuals histogram
    ax_3.set_title("Histogram of residuals", fontsize = subplot_title_size)
    sns.distplot(resids, ax = ax_3, kde = False);
    ax_3.set_xlabel(target_name, fontsize = subplot_label_size)
    ax_3.set_ylabel("Frequency", fontsize = subplot_label_size)

    plt.tight_layout() # handles most overlaping and spacing issues



def error_metrics(y_true, y_preds, n, k):
    '''
    Takes y_true, y_preds,  
    n: the number of observations.
    k: the number of independent variables, excluding the constant.
    Returns 6 error metrics
    '''
    def r2_adj(y_true, y_preds, n, k):
        rss = np.sum((y_true - y_preds)**2)
        null_model = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - rss/null_model
        r2_adj = 1 - ((1-r2)*(n-1))/(n-k-1)
        return r2_adj
    
    print('Mean Square Error: ', mean_squared_error(y_true, y_preds))
    print('Root Mean Square Error: ', np.sqrt(mean_squared_error(y_true, y_preds)))
    print('Mean absolute error: ', mean_absolute_error(y_true, y_preds))
    print('Median absolute error: ', median_absolute_error(y_true, y_preds))
    print('R^2 score:', r2_score(y_true, y_preds))
    print('Adjusted R^2 score:', r2_adj(y_true, y_preds, n, k))



def extract_individual_summary_table_statsmodel(X, y, table_number):
    '''
    Extracts individual summary table from statsmodel.summary
    Takes X_test, y_test, and table_number
    Returns a dataframe
    '''
    X = sm.add_constant(X)
    y = y
    model = sm.OLS(y,X).fit()
    summary_df = StringIO(model.summary().tables[table_number].as_csv())
    meta_df = pd.read_csv(summary_df)
    return meta_df




