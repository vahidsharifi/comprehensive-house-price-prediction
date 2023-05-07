import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class data_overview():

    
    # dscr is text file of feature description
    def __init__(self, df, dscr):
        self.df = df
        self.dscr = dscr
        


    def head_tail(self):
        """
        This function prints the five first and last rows of our dataframe
        """
        print('\n\n\n\n\nData Head:\n')
        display(self.df.head())
        print('\n\n\n\n\nData Tail:\n')
        display(self.df.tail())
    

    
    def features_description(self):
        """
        This function print the documentation of the dataset.
        """
        print('\n\n\n\n\nThe Description for Features:\n\n')
        print(self.dscr)
        

    
    def descriptive_statistics(self):
        """
        This function provides descriptive statistic of the different features."""
        i = 5
        j = 0
        while True:
            print(self.df.describe().iloc[:, j:i], '\n')
            if (i + 5) < self.df.describe().shape[1]:
                j += 5
                i += 5
            else:
                i = self.df.describe().shape[1]
                j += 5
                print(self.df.describe().iloc[:, j:i])
                break
    
        
    
    def data_information(self):
        """
        This function print information about the dataset such as the data type for each features,
        number of non-null values, memory used for dataset, etc.
        """
        print(self.df.info())
        print('\n')
    
    

    def features_with_null_values(self):
        """
        This function shows the columns contain null values
        , also it shows the number of null values within them.
        """
        null_included_features = self.df.isnull().sum()
        null_included_features = null_included_features[null_included_features != 0]
        if len(null_included_features) != 0:
            print('The features containing null values : \n')
            print('Feature     number of null\n')
            print(null_included_features)
        else:
            print('No feature contains null value')
    








class visualization():
    
    
   
    def __init__(self, df):
        self.df = df
        self.categorical_features_list = ['OverallQual', 'GarageCars', 'ExterQual', 'BsmtQual',
                                           'FullBath', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces']
    
    
    
    def shape(self):
        print(self.df.shape)
        

        
    def features_price_scatter(self):
        """
        This function visualize the scatter plot of the target ('SalePrice') and other non-categorical features.
        For instance if we have 'n' features, this will return 'n' different graphs.
        the y axis is price, and the x axis is the feature.
        """
        cnames = list(self.df.columns)
        
        # Removing categorical features from all features
        for feature in self.categorical_features_list:
            cnames.remove(feature)
        
        # Removing target from non-categorical features
        cnames.remove('SalePrice')
        
        # Specifying figure size
        plt.figure(figsize=(16, 50))
        
        # Create a color-map list for passing to graphs
        colors = ['#FFAEBC', '#A0E7E5', '#B4F8C8', '#FFA384', '#0C2D48', '#B1D4E0',
                  '#EF7C8E', '#FA26A0', '#FFAEBC', '#A0E7E5', '#B4F8C8', '#FFA384',
                  '#0C2D48', '#B1D4E0', '#EF7C8E', '#FA26A0']
        
        # Plotting the Features scatter
        for i, cname in enumerate(cnames):
            plt.subplot(int(np.ceil(len(self.df.columns) / 2)), 2, i+1)
            sns.scatterplot(self.df, x=cname, y='SalePrice', color=colors[i], edgecolor='gray')
            plt.title(f' Price vs. {cname}')
            plt.xlabel(f'{cname}')
            plt.ylabel('Price')
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8, rotation=30)
            
        plt.tight_layout(pad=5)
        plt.show()
        
        
        
    def price_distribution(self):
        """
        This function visualize the Price distribution and its outliers.
        It returns two graphs. One is the Histogram of the sale price.
        The other one is the BoxPlot.
        """
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        sns.histplot(self.df['SalePrice']) 
        plt.subplot(1, 2, 2)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel('SalePrice')
        plt.boxplot(self.df['SalePrice'])     
        
        
        
    def features_boxplots(self):
        """
        This function generate box plot for the features.
        It is good for see the distribution of different features in comparison with each others.
        """
        box_groups = [['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea', 'MasVnrArea'],
              ['YearBuilt', 'YearRemodAdd'], ['OverallQual', 'ExterQual', 'BsmtQual', 'KitchenQual'],
              ['FullBath', 'TotRmsAbvGrd', 'GarageCars', 'Fireplaces'] ]

        for group in box_groups:
            plt.figure(figsize=(2 * len(group), 1.7 *len(group)))
            self.df.boxplot(column=group)
            plt.show()
            print('\n')
            
            
            
    def pair_plots(self):
        """
        This function provide pair-scatter-plots for all un-categorical features.
        The diagonal of this function shows the distribution of each feature.
        Also the blue line in each graph represent the regression liine between two features.
        """
        sns.pairplot(self.df[['YearRemodAdd', 'MasVnrArea',
                         'OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
                         'GarageArea', 'TotRmsAbvGrd', 'SalePrice']], kind='reg',
                     plot_kws = {'scatter_kws':{'alpha':0.2, 'color':'#8a817c'}, 'line_kws':{'color':'#3a6ea5'}})
        plt.tight_layout()
        plt.show()
        
        
        
    def feature_price_boxplot(self):
        '''
        Box-plots to illustrate relationship between categorical features and SalePrice
        ''' 
        fig = plt.figure(figsize=(16,25))
        for i, feature in enumerate(self.categorical_features_list):
            ax = fig.add_subplot(4,2, i+1)
            sns.boxplot(x=feature, y='SalePrice', data=self.df, ax=ax)
        fig.tight_layout(pad=5)
        plt.show()
        

        
    def distribution_comparison(self, transformed_df, skew_dataframe):
        

        skewed_features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
                            'GarageArea', 'SalePrice', 'MasVnrArea']
        for c in skewed_features[:-1]:

            # Create a figure
            plt.figure(figsize=(14,5))

            # Creating a subplot
            plt.subplot(1,2,1)

            # Plotting the histogram
            sns.histplot(data=self.df, x=c, stat="density", alpha=0.4, kde=True, kde_kws={"cut": 3})

            # add title to subplot
            plt.title(f'Before transformation skew: {round(skew_dataframe.loc[c][0], 2)}')

            # the same as above
            plt.subplot(1,2,2)
            sns.histplot(data=transformed_df, x=c, stat="density", alpha=0.4, kde=True, kde_kws={"cut": 4}, color='orange')
            plt.title('After transformation' + f'{round(skew_dataframe.loc[c][1], 2)}')

            # showing the plot
            plt.show()
    
    