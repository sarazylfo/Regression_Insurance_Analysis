import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def drop_na_columns(dataframe, list_of_columns, threshold):
    """Drop columns where number of null entries in a column exceeds a user-set percentage threshold"""
    n = dataframe.shape[0]
    to_drop = [column for column in list_of_columns if (dataframe[column].isnull().sum() / n) > threshold]
    dataframe.drop(to_drop, axis = 1, inplace = True)
    print ('Number of dropped columns: {}'.format(len(to_drop)))
    print ('\n')
    print ('Dropped columns: \n', to_drop)
    
def categorical_and_discrete_na_filler(dataframe, categorical_columns):
    """Fill empty rows with values from selected column according to current distribution percentages"""
    for column in categorical_columns:
        choice = sorted(dataframe[dataframe[column].notnull()][column].unique())
        probability = dataframe[column].value_counts(normalize = True).sort_index().values
        dataframe[column] = dataframe[column].apply(
            lambda x: np.random.choice(choice, p = probability) 
            if (pd.isnull(x)) 
            else x)
        
def continuous_na_filler(dataframe, columns, method):
    """Fill empty rows with values according to user-chosen method; mean or median"""
    if method == 'mean':
        for column in columns:
            value = np.mean(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    elif method == 'median':
        for column in columns:
            value = np.nanmedian(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    else:
        print ('Method not available. Please choose either mean or median, else update function for desired method.')
        
def check_outliers(dataframe, list_of_columns, lower_quantile_list, upper_quantile_list):
    """Returns a dataframe of outliers according to user provided quantiles"""
    quantile = lower_quantile_list + upper_quantile_list

    summary_dict = {}
    for col in list_of_columns:
        summary_dict[col] = []
        for i in quantile:
            summary_dict[col].append(dataframe[col].quantile(i))

    summary_df = pd.DataFrame(summary_dict)
    summary_df_final = pd.concat([pd.DataFrame(quantile, columns=['Quantile']), summary_df], axis = 1)

    return summary_df_final

def drop_values_multi(dataframe, list_of_columns, quantile):
    """Drop outliers based on quantile """
    to_drop_index = []
    quantile = quantile

    for i in list_of_columns:
        index = list(dataframe[dataframe[i] > dataframe[i].quantile(quantile)].index)
        to_drop_index = to_drop_index + index

    dataframe.drop(set(to_drop_index), axis = 0, inplace = True)
    print ('Successfully dropped rows!')
    
    
def central_limit_mean(dataset, sample_size = 50, num_simulations = 500, return_mean = False):    
    """xxxxxx"""
    random_chosen = [np.mean(np.random.choice(dataset, size = sample_size)) for i in range(num_simulations)]
    if return_mean == False:
        return random_chosen
    else:
        return (random_chosen, round(np.mean(random_chosen), 2))
    
def CLT_violinplots(dataframe, x_axis, y_axis, sample_size = 50, num_simulations = 500):
    """xxxxxx"""
    unique_list = dataframe[y_axis].unique()
    df = pd.DataFrame(None, columns = ['{} types'.format(y_axis), '{} Sample Mean'.format(x_axis)])
    
    for i in unique_list:
        CLT_data = central_limit_mean(dataframe[dataframe[y_axis] == i][x_axis], 
                                      sample_size = sample_size, 
                                      num_simulations = num_simulations)
        df = pd.concat([df, pd.DataFrame(list(zip([i] * len(CLT_data), CLT_data)), 
                                         columns = [y_axis, '{} Sample Means'.format(x_axis)])], 
                        axis = 0, 
                        ignore_index=True)
                        
    ordering = df.groupby(y_axis)['{} Sample Means'.format(x_axis)].mean().sort_values().index
        
#   plt.figure(figsize=(15,10))
    sns.set(font_scale = 1.1)
    sns.violinplot(x = '{} Sample Means'.format(x_axis), y = y_axis, data = df, palette="Set3", order = ordering)
    plt.xticks(rotation = 45)
    plt.show()