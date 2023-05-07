import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score
from sklearn import preprocessing
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor 





class machine_learning():
    
    
    def __init__(self, df, transformed_df):
        
        self.features, self.target = feature_target_splitter(df)
        
        # t_features, t_target : features and target for transformed dataframe
        self.t_features, self.t_target = feature_target_splitter(transformed_df)
    
    
    
    def multi_regr_with_scikit(self, num_iteration=50):
        
        """
        This function uses Scikit-Learn Multivariate Liniear Regression model to perform
        linear regression in our database.
        It takes two dataframe as the input.
        It prints the accuracy for both dataframes.
        Also it returns a dataframe which contains three column.
        
        
        coef : the coeficient for different features for transformed model with random_stae=50.
        average : the mean value for each feature on the dataframe
        Effect on price : Effect on price = coef * average. This index is a measurment to evaluate the effect of each feature on price.
        the bigger absolute value of this number, the more effect this feature has on price.
    
        
        As our model accuracy depends on how we splited data to trian and test,
        this function perform linear regression for 50 different train and test datasets.
        Then it return the average of all models to prevent the biased result regarding train and test random_seed.
        
        
        num_iteration : The number of times we perform linear regression with different train and tests.
        df : Original dataframe without data transformation
        transformed_df : The dataframe with transformed values
        Then do linear regression separately for both these datasets.
        """
        
        # defining two variables for storing the accuracy of models
        normal_model = 0
        transformed_model = 0
        
        # Create a for loop for run on model for num_iteration dufferent train, test datasets
        for i in range(int(num_iteration)):
            
            # splitting the original and transformed datasets to train and test datasets
            x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=i)
            xt_train, xt_test, yt_train, yt_test = train_test_split(self.t_features,
                                                                                     self.t_target, random_state=i)
            # building our model for original dataset
            model = LinearRegression()
            
            # fitting the regression on our model
            model.fit(x_train, y_train)
            
            # adding the model accuracy to score container
            normal_model = normal_model + model.score(x_test, y_test)

            # Same steps as above for transformed dataset
            model = LinearRegression()
            model.fit(xt_train, yt_train)
            transformed_model = transformed_model +  model.score(xt_test, yt_test)

        # deviding the scores by number of iterations to get the average r-squared
        normal_score = normal_model / num_iteration
        transformed_score = transformed_model / num_iteration
        
        # creating a dataframe to store the coeficients and average values of features
        data = pd.DataFrame({'coef':np.around(model.coef_, 4),
                             'average':[self.features[i].mean() for i in self.features.columns]}, index=self.features.columns)
        
        # evaluating features effect on price
        data['Effect on price'] = [data.iloc[i ,0] * data.iloc[i, 1] for i in range(len(data.index))]
        
        # printing the average accuracy for original and transformed datasets
        print('Normal model score is: ', normal_score)
        print('Transformed model score is: ', transformed_score, '\n\n')
        
        return data
    
    
    
    
    
    
    def multi_regr_with_lstat(self, num_iteration=50):
        
        """
        This function uses StatsModel Ordinary Least Squares Liniear Regression model to perform
        multivariate linear regression on our database.
        It takes two dataframe as the input.
        It prints the accuracy for both dataframes.
        Also it returns a dataframe which contains three column.
        
        
        coef : the coeficient for different features for transformed model with random_stae=50.
        average : the mean value for each feature in the dataframe
        Effect on price : Effect on price = coef * average. This index is a measurment to evaluate the effect of each feature on price.
        the bigger absolute value of this number, the more effect this feature has on price.
    
        
        As our model accuracy depends on how we splited data to trian and test,
        this function perform linear regression for 50 different train and test datasets.
        Then it return the average of all models to prevent the biased result regarding train and test random_seed.
        
        
        num_iteration : The number of times we perform linear regression with different train and tests.
        df : Original dataframe without data transformation
        transformed_df : The dataframe with transformed values
        Then do linear regression separately for both these datasets.
        
        Outputs:
        
        """

        # defining two variables for storing the accuracy of models
        normal_model = 0
        transformed_model = 0

        # Create a for loop for run on model for num_iteration dufferent train, test datasets
        for i in range(int(num_iteration)):

            # splitting the original and transformed datasets to train and test datasets
            x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=i)
            xt_train, xt_test, yt_train, yt_test = train_test_split(self.t_features,
                                                                                     self.t_target, random_state=i)
            
            #create a dataframe with a column name constant
            x_incl_const = sm.add_constant(x_train) 
            
            # building our model for original dataset
            model = sm.OLS(y_train, x_incl_const)
            
            # fitting the regression on our model and store the result in results variable
            results = model.fit() 
            normal_model += results.rsquared
            
            
            prices = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})

            
            # Same steps as above for transformed dataset
            x_incl_const = sm.add_constant(xt_train) 
            model = sm.OLS(yt_train, x_incl_const) 
            results = model.fit()
            transformed_model += results.rsquared
            
            
            # creating a dataframe to store the coeficients and p-values of features
            short_summary = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})
            
            # VIF Factor column to store variance inflation factor for each column
            vif = [round(variance_inflation_factor(x_train.values, i),3)  for i in range(x_train.shape[1])]
            vif.insert(0, 0)
            short_summary["VIF Factor"] = vif
            long_summary = results.summary()
    
        # Printing BIC ,AIC, and r-squared for transformed model with random_state = 50
        print('BIC is: ' , results.bic)
        print('AIC is: ' , results.aic)
        print('r-squared for original dataframe is: ',normal_model/num_iteration)
        print('r-squared for modified dataframe is: ',transformed_model/num_iteration)


        return short_summary, long_summary



    


    def multi_regr_with_randomforest(self, num_iteration=50):

        # defining two variables for storing the accuracy of models
        normal_model = 0
        transformed_model = 0
        log_evs = 0
    #     prediction_df = pd.DataFrame(index=range(len(test_features)), columns=range(int(num_iteration)))


        # Create a for loop for run on model for num_iteration dufferent train, test datasets
        for i in range(int(num_iteration)):
            
            # splitting the original and transformed datasets to train and test datasets
            x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=i)
            xt_train, xt_test, yt_train, yt_test = train_test_split(self.t_features,
                                                                                     self.t_target, random_state=i)

            # building our model for original dataset
            model = RandomForestRegressor(n_estimators = 10, random_state = 0)
            
            # fitting the regression on our model 
            model.fit(x_train, y_train) 
            
            # Storing the prediction for test dataset in y_pred 
            y_pred = model.predict(x_test)
            
            # adding the r2-squared (accuracy) to the container
            normal_model = normal_model + r2_score(y_test, y_pred)


            # Same steps as above for transformed dataset
            model = RandomForestRegressor(n_estimators = 10, random_state = 0)
            model.fit(xt_train, yt_train)
            yt_pred = model.predict(xt_test)
            transformed_model = transformed_model + r2_score(yt_test, yt_pred)

        # deviding the scores by number of iterations to get the average r-squared
        normal_score = normal_model / num_iteration
        transformed_score = transformed_model / num_iteration
        
        # printing the average accuracy for original and transformed datasets
        print('Normal model score is: ', normal_score)
        print('Transformed model score is: ', transformed_score)
   

    


    
def feature_target_splitter(df):
        """
        This function takes the original dataframe and returns features and targets
        """ 
        # Creating Features and Targets
        target = df['SalePrice']
        features = df.drop('SalePrice', axis=1)
        
        return features, target
    
        
