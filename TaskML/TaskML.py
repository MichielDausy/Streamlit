# %% [markdown]
# # To Do
#
# - Explain the 2 ML techniques
# - Explain the confusion matrix
# - Compare the 3 algorithms
# - Host on streamlit

# %% [markdown]
# ## Connect-4 dataset
# I chose a dataset that contains all legal 8-ply positions in the game of connect-4 in which neither player has won yet, and in which the next move is not forced.
#
# There are 2 players (X and O), the outcome of the match is either win, lose or draw for player X.
# I chose this because I already know what connect-4 is and how it works so it will be easier to interpret and understand the data compared to a dataset that is about a topic where I barely know anything about.

# %% [markdown]
# ## What is an EDA?
# Exploratory Data Analysis (EDA) is a method used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions. EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a better understanding of data set variables and the relationships between them

# %% [markdown]
# ### There are 8 steps in performing an EDA:
# - Import Libraries and Load Dataset: importing necessary libraries such as pandas, numpy, graphviz, etc., and load your dataset.
# - Check for Missing Values
# - Visualizing the Missing Values: Use visual techniques to identify where the missing values are located
# - Replacing the Missing Values: Decide on a strategy to handle these missing values
# - Asking Analytical Questions and Visualizations: Formulate questions you want to answer from the dataset and use visualizations to find these answers
# - Data Cleaning/Wrangling: Clean the data by removing duplicates, handling outliers, etc
# - Feature Engineering: Explore various variables and their transformations to create new features or derive meaningful insights
# - Statistics Summary: Generate summary statistics for numerical data in the dataset

# %% [markdown]
# #### Import Libraries and Load Dataset
# The first step is to install and import the packages and load the dataset.

# %%
""" !pip install graphviz
!pip install pydotplus
!pip install pandas
!pip install seaborn
!pip install streamlit

# %%
#pip install scikit-learn

# %%
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# %%
# load dataset
connect_df = pd.read_csv("connect-4.data/connect-4.data", sep=',')

# %% [markdown]
# I use the describe function to give a summary of the central tendency, dispersion, and shape of the distribution of a dataset, excluding NaN values.
# The describe function didn't show all the columns so I had to define this manually to use 43 columns because the dataset has 43 columns.

# %%
with pd.option_context('display.max_columns', 43):
    print(connect_df.describe())

# %% [markdown]
# Here I get my first problem where the dataset didn't contain column names. This is a problem because the describe function takes the top row to define the columns of the dataset. In this case the top row also has values: [b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win] so I have to change the dataset to include column names.

# %% [markdown]
# The dataset was originally made with this layout for the connect-4 playboard where the possible spots are represented like cöordinates:

# %% [markdown]
# <img src="resources/Playboard.png" style="height: 300px"/>

# %% [markdown]
# So I applied these cöordinates to the names of the columns in the updated dataset:

# %%
# load updated dataset
updatedConnect_df = pd.read_csv("connect-4.data/connect-4-with-column-names.data", sep=',')

with pd.option_context('display.max_columns', 43):
    print(updatedConnect_df.describe())

# %% [markdown]
# As you can see I only get 4 statistics back because the dataset contains String values.
# the 4 statistics are describes as such:
# - Count: Number of non-null (or, non-NaN) observations. This gives you the total number of non-null entries in the column.
#     - In this case we have 67556 rows + 1 row for the column names for each column. This also confirms that there are no missing values because the amount of rows in the dataset is also 67556.
# - Unique: Number of distinct values in a column. This tells you how many different categories or labels are present in the column.
#     - There are 3 unique values for each possible cöordinate on the board:
#         - b = blank
#         - x = player x has taken this cöordinate
#         - o = player o has taken this cöordinate
# - Top: Most frequent value. This is the value that appears most often in the column.
#     - for the most part the blank option is the most used in each row except for row A6 and B6.
# - Freq: Frequency of the most frequent value. This tells you how many times the most frequent value appears in the column.

# %% [markdown]
# ### Show the data on the board
# 
# In this code you can see how a row in the dataset represents a state on the board.

# %%
# Select a row from the DataFrame
selected_row = updatedConnect_df.iloc[10]  # You can change this to select other rows

# Now I convert the selected row into a 7x6 numpy array
board = np.array(selected_row[:-1]).reshape(7, 6)

# Transpose the board
board = board.T
# Reverse the rows
board = board[::-1]

# Print the board
print(board)

# %% [markdown]
# ## Split the data
# 
# Given the cöordinates I will predict if the outcome will result in a win a loss or a draw. N?ow to split the features and the target variable:

# %%
# split dataset in features and target variable

feature_cols = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','B4','B5','B6','C1','C2','C3','C4','C5','C6','D1','D2','D3','D4','D5','D6','E1','E2','E3','E4','E5','E6','F1','F2','F3','F4','F5','F6','G1','G2','G3','G4','G5','G6']

X = updatedConnect_df[feature_cols]
y = updatedConnect_df[['Outcome']] # target variable

# %%
print(X)

# %%
print(y)

# %% [markdown]
# Here I replace the categories with numerical values using `One-Hot encoding`. With one-hot encoding, we convert each categorical value into a new categorical value and assign a binary value of 1 or 0.

# %% [markdown]
# Firstly I need to train the machine learning techniques. To do this I first need to install `category_encoders` to replace the categories with numeric values because **the decision trees implemented in scikit-learn use only numerical features and these features are interpreted always as continuous numeric variables.**

# %%
#pip install --upgrade category_encoders

# %%
import category_encoders as ce

ce_oh = ce.OneHotEncoder(cols = feature_cols)
X_cat_oh = ce_oh.fit_transform(X)

# %% [markdown]
# What the encoder does is put a 1 on the spot where it sees something is 'filled in'. every feature is splitted into 3 columns because there are 3 possible values for each feature. A1 becomes [A1_1, A1_2, A1_3], these 3 values represent the possible values [b, o, x] and if for example b is filled in in A1 then A1_1 gets the value 1.

# %%
print(X_cat_oh)

# %% [markdown]
# The dataset still needs to be splitted into training data and testing data. The process of splitting the data involves randomly assigning data points to either the training set or the test set. A common split ratio is 80% for training and 20% for testing, but this can vary depending on the size and nature of a dataset.

# %% [markdown]
# Here I split 20% of the data into test data and 80% into training data. I also use a seed for the random number generator used for the split.

# %%
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cat_oh, y, test_size=0.2, random_state=42)

# %% [markdown]
# These are the labels for the confusion matrix

# %%
labels = ['draw', 'lose', 'win']

# %% [markdown]
# ## Decision Tree
# 
# For the first ML learning algorithm I will use the decision tree.
# 
# A decision tree is capable of performing classification on a dataset. It takes 2 arrays as input:
# - Array X, these are the training samples so in this case the positions players have taken on the board.
# - Array Y, these are the labels for the training samples so in this case the outcome of the sample (win, lose, draw).

# %% [markdown]
# Now we can try to train the classifier (decision tree) with the training data.

# %%
clf = DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(X_train, y_train)

# %% [markdown]
# Now we can make predictions using the test data.

# %%
predictions = clf.predict(X_test)

# %% [markdown]
# And then calculate the accuracy of the algorithm using the actual target values of your test data and compare them with the predicted values.

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", round(accuracy, 2) * 100, "%")

# compute the confusion matrix
cmTree = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cmTree, display_labels=labels).plot()

# %% [markdown]
# In the graph above you can see when the model predicted right and when it went wrong.
# 
# The labels on the left are what the outcome of the test data actually was whilst the labels on the bottom are the predictions the model made.
# 
# So for example with the decision tree if a sample had an outcome 'draw' the model would predict the right outcome 391 times (top left square).
# 
# The other times it would predict 'draw' 382 times for an outcome that was supposed to be 'lose'.
# 
# From this confusion matrix we can see that it got a lot of predictions right when it came to the outcome 'win', this is because there are a lot more samples in the dataset where the actual outcome is 'win' so it was trained better to predict that outcome.

# %% [markdown]
# ## Support Vector Machine
# 
# Support Vector Machine (SVM) is a machine learning algorithm used for linear or nonlinear classification, regression, and even outlier detection tasks, but it is best suited for classification.
# SVMs are adaptable and efficient in a variety of applications because they can manage high-dimensional data.
# 
# What the SVM algorithm does is find the optimal **hyperplane** in a N-dimensional space that separates the data in different classes. In other words the distance between the closest data points from 2 different classes should be as big as possible.
# The hyperplane serves as a decision boundary to seperate all the data points of different classes in the feature space.
# 
# The dimension of the hyperplane depends on the amount of features in the dataset, here it will be a 42-dimensional hyperplane.
# 
# ### How does it work with an example
# 
# In a dataset with 2 features the hyperplane is a line because 2 features means it is 2 dimensional. Let's say we have 2 classes in the dataset represented as blue and red in the image:
# 

# %% [markdown]
# <img src="resources/svm1.svg.png" style="height: 300px">

# %% [markdown]
# The algorithm then chooses which hyperplane is located the furthest away from both the nearest data points from the red and blue classes. This hyperplane is also known as the **maximum-margin hyperplane**.
# 
# In the image above the best **maximum-margin hyperplane** is clearly "A" because the distance from the 2 nearest data points from both red and blue is greater then the distance from the 2 nearest data points from both red and blue on hyperplane "B".

# %% [markdown]
# ### Outliers
# 
# What if a red data point is located under the hyperplane (so in the area of the blue data points)?
# 
# In this case the maximum-margin hyperplane is located the same way as normal, but a penalty is given each time a data point crosses the hyperplane. so here a red point in the blue area gives a penalty.

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Create a scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

svm_clf = SVC()
svm_clf.fit(X_train_scaled, np.ravel(y_train))

predictions = svm_clf.predict(X_test_scaled)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", round(accuracy, 2) * 100, "%")

# compute the confusion matrix
cmVector = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cmVector, display_labels=labels).plot()

# %% [markdown]
# ## Stochastic Gradient Descent
# 
# Stochastic Gradient Descent (SGD) is a variant of the Gradient Descent algorithm that is used for optimizing machine learning models. It is more efficient when dealing with larger datasets compared to the regular Gradient Descent algorithm.
# 
# In SGD, instead of using the entire dataset for each iteration, only a single random training example or batch is selected to calculate the gradient and update the model parameters.

# %% [markdown]
# ### How does it work
# 
# there are 4 steps in the algorithm:
# - Initialization: first the parameters of the model are randomly initialized.
# - Set parameters: then the parameters are set by defining the number of iterations and the learning rate.
# - Stochastic Gradient descent loop: here there are  5 more steps:
#     - The training dataset is shuffled to achieve randomness
#     - Then we iterate over each training sample or batch
#     - Compute the gradient of the cost function with respect to the model parameters using the current training example (or batch).
#     - Update the model parameters by taking a step in the direction of the negative gradient, scaled by the learning rate.
#     - Evaluate the convergence criteria, such as the difference in the cost function between iterations of the gradient.
# - Return optimized parameters: Once the convergence criteria are met or the maximum number of iterations is reached, return the optimized model parameters.

# %%
from sklearn.linear_model import SGDClassifier

lr_clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=100)
lr_clf.fit(X_train, np.ravel(y_train))

predictions = lr_clf.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", round(accuracy, 2) * 100, "%")

# compute the confusion matrix
cmGradient = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cmGradient, display_labels=labels).plot()

# %% [markdown]
# ## Comparison

# %%
ConfusionMatrixDisplay(cmTree, display_labels=labels).plot()

ConfusionMatrixDisplay(cmVector, display_labels=labels).plot()

ConfusionMatrixDisplay(cmGradient, display_labels=labels).plot() """

# %% [markdown]
# ## Streamlit

# %%
import pandas as pd
import streamlit as st
import numpy as np

# load updated dataset
updatedConnect_df = pd.read_csv(
    "TaskML/connect-4.data/connect-4-with-column-names.data", sep=","
)

st.title('ML Algorithm Performance')

st.write("EDA of dataset:")
with pd.option_context('display.max_columns', 43):
    st.write(updatedConnect_df.describe())

# Add a slider to select the row number
row_number = st.slider('Select a board state:', min_value=0, max_value=len(updatedConnect_df)-1, value=10)

# Select a row from the DataFrame
selected_row = updatedConnect_df.iloc[row_number]  # You can change this to select other rows

# Now I convert the selected row into a 7x6 numpy array
board = np.array(selected_row[:-1]).reshape(7, 6)

# Transpose the board
board = board.T
# Reverse the rows
board = board[::-1]

# Convert the board to a DataFrame and replace the column names
board_df = pd.DataFrame(board, columns=list('ABCDEFG'), index=range(6, 0, -1))

# Print the board
st.write("Connect-4 board:")
st.table(board_df)

# split dataset in features and target variable

feature_cols = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
]

X = updatedConnect_df[feature_cols]
y = updatedConnect_df[["Outcome"]]  # target variable

import category_encoders as ce

ce_oh = ce.OneHotEncoder(cols=feature_cols)
X_cat_oh = ce_oh.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_cat_oh, y, test_size=0.2, random_state=42
)


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def get_metrics(_clf, X_train, y_train, X_test, y_test):
    _clf.fit(X_train, np.ravel(y_train))
    y_pred = _clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return cm, acc

def plot_confusion_matrix(cm, clf_name):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues', ax=ax)
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')
    ax.set_title(clf_name)
    labels = ['Draw', 'Lose', 'Win']  # Define your own labels here
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    return fig

# Define the classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machines': SVC(),
    'Stochastic Gradient Descent': SGDClassifier()
}

# Create a multiselect box for selecting the classifiers
selected_clfs = st.multiselect('Select Classifier(s)', list(classifiers.keys()))

# Initialize session state for storing classifier results
if 'clf_results' not in st.session_state:
    st.session_state['clf_results'] = {}

# Calculate metrics and plot confusion matrix for each selected classifier
for clf_name in selected_clfs:
    if clf_name not in st.session_state['clf_results']:
        clf = classifiers[clf_name]
        cm, acc = get_metrics(clf, X_train, y_train, X_test, y_test)
        st.session_state['clf_results'][clf_name] = (cm, acc)

    st.header(clf_name)
    cm, acc = st.session_state['clf_results'][clf_name]
    st.write(f'Accuracy: ', round(acc, 4) * 100, "%")
    fig = plot_confusion_matrix(cm, clf_name)
    st.pyplot(fig)

# %% [markdown]
# ## Generative AI Tools
# ### Prompts used
# - What is an EDA and how do I perform this on a dataset? - BingAI - with this prompt a very long explanation was given about EDA followed by the steps that need to be performed in that process.
#
# - Using the pandas library explain what describe does - BingAI - I get an explanation of all the statistics from the describe function output and the function is also shown iin an example but the statistics aren't the same as mine because i have a dataset with String values so I had to ask further:
#     - I apply this function on a dataset with string values so explain the statistics from that result - bingAI - Now it gives me the right statisticsand explains them.
#
# - Briefly explain the ML learning alogrithm for a decision tree - BingAI - This gives a short explanation of the algorithm with an example, in this example it also shows how to divide the dataset into training data and test data so I asked more information about this:
#     - explain how the splitting of the data into training data and test data works - BingAI - Here it gives a reason why data is splitted into test and training data and it also says that usually 80% of the data is training data and 20% is test data.
#
# - Explain what one hot encoding is - BingAI - Here it gives a complicated explanation of what one-hot encoding is.
#
# - show how I can make an accuracy of the decision tree when i'm using one hot encoding - BingAI - Here it gives example code showing how I can measure the accuracy of the algorithm, but it only showed how to get the accuracy so I asked if there were more measurements possible:
#     - are there more ways of measuring the ML algorithm so i can compare it to other algorityhms? - BingAI - It then gives different ways for measuring ML algorithms for classification, regression, clustering and ranking.
