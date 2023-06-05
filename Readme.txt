
macOS preparation Instruction:
1 - Open a terminal window at the py_project folder2 - Create a virtual environment using the following command:
python3 -m venv venv 3 -  Activate the virtual environment in your terminal using the following command:
source venv/bin/activate4 - Install requirements and dependencies in your virtual environment using the following command:
pip install -r requirements.txt6 - Open Visual Studio Code application and open py_project folder

7 - Right-click on the explorer sidebar and click on open in integrated terminal.

8 - click on the main.ipynb and run it.Note: Depending on the visual studio's extensions and kernels you've previously installed and created, although you activated the current virtual environment, the Visual Studio Code might not automatically switch the kernel to the current environment. So while you want to execute the main.ipynb, your VScode might offer you to choose kernel source. In this case, follow these instructions:
1. Once you run the cell a box might pops up asking to choose the source kernel. Choose Pythone environments.
2. Choose venv Python

*** Please note that due to the hard installation procedure of TensorFlow on macbooks with M series processors, we don't put TensorFlow library in the requirements.txt file as it will not be installed. Consequently, you can not run ANN regression on your virtual environment. Nonetheless, using the link below, you can run this project online on the Kaggle website. Please note that the local file is developed version of kaggle. So use this link for checking the result of ANN regressor.Link: https://www.kaggle.com/code/vahidsharifi76/house-price-python-uogLibraries:
# seaborn: For advanced data visualization and plotting, enhancing the visual representation of data.
# numpy: For numerical computations, efficient handling of arrays, and mathematical operations.
# pandas: For data manipulation, analysis, and working with structured data in tabular form.
# matplotlib.pyplot: For creating plots, charts, and visualizations of data.
# seaborn: Additional functionality for data visualization, complementing matplotlib.
# os: For interacting with the operating system, such as file and directory manipulation.
# scipy.stats.boxcox: For performing the Box-Cox transformation, a data transformation technique that helps normalize skewed data.
# statsmodels.api: For advanced statistical models and analysis, providing a wide range of statistical tools.
# statsmodels.stats.outliers_influence.variance_inflation_factor: For detecting multicollinearity, a measure of correlation between predictor variables.
# sklearn.preprocessing.StandardScaler: For standardizing feature scaling, ensuring each feature has a mean of 0 and standard deviation of 1.
# sklearn.model_selection.train_test_split: For splitting data into training and testing sets, evaluating model performance.
# sklearn.linear_model.LinearRegression: For implementing linear regression models, fitting a linear equation to the observed data.
# sklearn.ensemble.RandomForestRegressor: For implementing random forest regression models, using an ensemble of decision trees.
# sklearn.metrics.r2_score: For evaluating regression model performance, measuring the amount of variance explained by the model.
# sklearn.metrics.explained_variance_score: For evaluating regression model performance, measuring the amount of variance explained by the model.
# sklearn.impute.KNNImputer: For imputing missing values using the K-Nearest Neighbors algorithm.
# sklearn.preprocessing: For preprocessing and data transformation, including scaling, encoding, and normalization.
# sklearn.metrics: For evaluating model performance and metrics, such as accuracy, precision, and recall.
# missingno: For visualizing missing data patterns in the dataset, identifying missing data and their distribution.

