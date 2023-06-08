This project is a House Price Prediction covering different aspects, such as Python Programming Skills,
Data Cleaning, Manipulation, Data Visualisation, Machine Learning, and Statistical Data Analysis techniques.
The project code is distributed into four files.
1- eda.py : It contains two main classes, one for descriptive analysis and another for data visualisation. 
2- manipulation.py : It contains classes and functions useful for feature transformation, statistical processing, and data manipulation.
3- machine_learning : It contains classes and functions for implementing machine learning algorithms.
4- main.ipynb : This is the main project resulting from all the code we programmed. It also contains an analysis of our code, also the dataset.

Note: It is good to mention that the 4000 words are distributed to all these files.

Note: As the TensorFlow library is not in requirements.txt, the final parts of the project related to ANN regression would not work on your local machine. So to run those parts, please refer to the link below and run it on the Kaggle.
 Link: https://www.kaggle.com/code/vahidsharifi76/house-price-python-uog

Dataset:
House Prices - Advanced Regression Techniques
Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

This dataset belongs to a competition on Kaggle. This competition contains two datasets, one is the training dataset which is for training the model. Another one is the test dataset.
However, for this study, we just used the training dataset as the aim of the study is different.  Our training dataset contains 1460 rows and 81 features, including the target column.
It's a dataset with a huge range of features containing strings and numerics.





macOS Preparation Instruction:


1 - Open a terminal window at the py_project folder


2 - Create a virtual environment using the following command:

python3 -m venv venv 


3 -  Activate the virtual environment in your terminal using the following command:

source venv/bin/activate


4 - Install requirements and dependencies in your virtual environment using the following command:

pip install -r requirements.txt


5 - In Visual Studio Code application, open py_project folder

6 - Right-click on the explorer sidebar and click on open in integrated terminal.

7 - Once again run this command:

source venv/bin/activate


8 - click on the main.ipynb and run it.

Note: Depending on the visual studio's extensions and kernels you've previously installed and created, although you activated the current virtual environment,
the Visual Studio Code might not automatically switch the kernel to the current environment. 
So while you want to execute the main.ipynb, your VScode might offer you to choose a kernel source. In this case, follow these instructions:
1. Install extensions visual studio offers, such as Jupyter kernel.
2. Once you run the cell, a box asking you to choose the source kernel might pop up. Choose Python environments which are associated with the current path.


*** Please note that due to the hard installation procedure of TensorFlow on Macbooks with M series processors,
 we don't put the TensorFlow library in the requirements.txt file as it will not be installed.
 Consequently, you can not run ANN regression on your virtual environment. Nonetheless, you can run this project online on the Kaggle website using the link below.
 Please note that the local file is a developed version of Kaggle. So use this link to check the result of the ANN regressor.
Link: https://www.kaggle.com/code/vahidsharifi76/house-price-python-uog





Windows 10 Instruction:

1 - Open a Windows Command Line at the py_project folder

2 - Create a virtual environment using the following command:

python -m venv venv 


3 -  Activate the virtual environment in your Windows command line using the following prompt:

venv\Scripts\activate


4 - Install requirements and dependencies in your virtual environment using the following command:

pip install -r requirements.txt


5 - In the Visual Studio Code application, open the py_project folder

6 - Right-click on the explorer sidebar and click on open in integrated terminal.

7 - Once again, run this command:

source venv/bin/activate


8 - click on the main.ipynb and run it.

Note: Depending on the visual studio's extensions and kernels you've previously installed and created, although you activated the current virtual environment,
the Visual Studio Code might not automatically switch the kernel to the current environment. 
So while you want to execute the main.ipynb, your VScode might offer you to choose a kernel source. In this case, follow these instructions:
1. Install extensions visual studio offers such as Jupyter kernel.
2. Once you run the cell, a box asking you to choose the source kernel might pop up. Choose Python environments which is associated with the current path.










Libraries:
# seaborn: For advanced data visualisation and plotting, enhancing the visual representation of data.
# numpy: For numerical computations, efficient handling of arrays, and mathematical operations.
# pandas: For data manipulation, analysis, and working with structured data in tabular form.
# matplotlib.pyplot: For creating plots, charts, and visualisations of data.
# seaborn: Additional functionality for data visualisation, complementing matplotlib.
# os: For interacting with the operating system, such as file and directory manipulation.
# scipy.stats.boxcox: For performing the Box-Cox transformation, a data transformation technique that helps normalize skewed data.
# statsmodels.api: Providing a wide range of statistical tools for advanced statistical models and analysis.
# statsmodels.stats.outliers_influence.variance_inflation_factor: For detecting multicollinearity, a measure of correlation between predictor variables.
# sklearn.preprocessing.StandardScaler: For standardising feature scaling, ensure each feature has a mean of 0 and standard deviation of 1.
# sklearn.model_selection.train_test_split: For splitting data into training and testing sets, evaluating model performance.
# sklearn.linear_model.LinearRegression: For implementing linear regression models, fitting a linear equation to the observed data.
# sklearn.ensemble.RandomForestRegressor: For implementing random forest regression models, using an ensemble of decision trees.
# sklearn.metrics.r2_score: For evaluating regression model performance, measuring the amount of variance explained by the model.
# sklearn.metrics.explained_variance_score: For evaluating regression model performance, measuring the amount of variance explained by the model.
# sklearn.impute.KNNImputer: For imputing missing values using the K-Nearest Neighbors algorithm.
# sklearn.preprocessing: For preprocessing and data transformation, including scaling, encoding, and normalization.
# sklearn.metrics: For evaluating model performance and metrics, such as accuracy, precision, and recall.
# missingno: For visualising missing data patterns in the dataset, identifying missing data and their distribution.


