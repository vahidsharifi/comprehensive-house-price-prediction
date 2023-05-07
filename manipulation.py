import pandas as pd
from sklearn import preprocessing
from scipy.stats import boxcox
import scipy.stats as stats





class preprocesing():
    

    def string_to_categorical(slef, df):
        """
        This funtion gets your dataframe, converts the string data to categorical data,
        and returns the modified dataframe as the output.

        """
        

        # Create a dataframe for store the changes
        modified_df = df
        # Selecting all columns with string data
        new_df = df.select_dtypes(include=['object'])

        for cname in new_df.columns:
            modified_df[cname] = pd.Categorical(modified_df[cname])

        return modified_df
    
    
    
    def categorical_to_numeric(self, df):
        """ 
        This function gets your dataframe, converts categorical data to numeric,
        and return modified dataframe and the dictionary of labels
        """
        # Creating label encoder object
        le = preprocessing.LabelEncoder()

        # Creating a new dataframe with just categorical features
        new_df = df.select_dtypes(include=['category'])
        # Create a dataframe for store the changes
        modified_df = df
        # Creating a dictionary for storing the labels
        col_dic = {}

        for cname in new_df.columns:
            series = df[cname]
            df[cname] = pd.Series(le.fit_transform(series[series.notnull()]),
                                     index=series[series.notnull()].index)
            le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
            col_dic[cname] = le_name_mapping

        return modified_df, col_dic
    



    
def feature_select(df, target):
    
    """
    This function takes the data frame, and return a dictionary of features
    with their correlations which all have correlations above +0.45 or blew -0.45
    
    df : dataframe name
    target: name of the column in dataframe containing the target values
    """
    # Storing the correlation between target and the other features
    correlations = [df[target].corr(df[i]) for i in df.columns]
    col_nr = 0
    corr_dic = {}

    for corr in correlations:
        if abs(corr) > 0.45:
            corr_dic[df.columns[col_nr]] = corr
        col_nr += 1
    
    return corr_dic





def box_cox_transformer(df):
    """
    This Function take the dataframe as the input and deploy box-cox transformation
    on the columns, then returns the transformed dataset, a pandas dataframe represents
    the skewness of the columns before and after the transformation, and a dictionary
    that contains the lambdas for different features for retransformation.
     """
    transformed_df = df.copy()
    skewed_features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea', 'SalePrice', 'MasVnrArea']
    initial_skew_list = []
    transformed_skew_list = []

    # Creating a dictionary to store lambdas in it
    lambda_dic = {}

    # Adding epsilon to data
    epsilon = 10e-10
    transformed_df = transformed_df + epsilon

    # Implementing box-cox
    for column in skewed_features:
        transformed_df[column], lambda_dic[column] = boxcox(transformed_df[column])
        initial_skew_list.append(df[column].skew())
        transformed_skew_list.append(transformed_df[column].skew())

    # Creating a dataframe to compare skewness before and after transformation
    skew_dataframe = pd.DataFrame(index=skewed_features, data={'Original Data Skew':initial_skew_list, 'Transformed Data Skew':transformed_skew_list} )
    
    return transformed_df, skew_dataframe, lambda_dic





def outlayer_remover(df):
    """
    This funtion get the dataframe, then drop the rows with target's standard deviation
    over the 3 or -3. Finally it returns the modified dataframe withot outlayers. 
    """
    # Create a copy of original daframe
    df_without_outlayers = df.copy()

    # Calculating z-scores for sale prices
    zscores = stats.zscore(df_without_outlayers['SalePrice'])

    # Capturing the outlayers
    outlayers = zscores[abs(zscores) > 3]

    # Removving outlayers from the daaset
    for index in outlayers.index:
        df_without_outlayers.drop(index, inplace=True)

    # Print the number of outlayers
    print(f'{df.shape[0] - df_without_outlayers.shape[0]} Houses are detected as outlayers')

    return df_without_outlayers