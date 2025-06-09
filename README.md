# Python-for-Data-Science

Learning Python from Jupyter Notebook and Visual Studio Code.

## 1 Jupyter Notebook .ipynb

### import necessary Python libraries
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- import numpy as np
- import statsmodels.api as sm

- from sklearn.model_selection import train_test_split
- from sklearn.neighbors import KNeighborsClassifier
- from statsmodels.formula.api import ols

### Basic Commands 
- print("Hello World")
- pd.read_csv("data.csv") # Load dataset into a pandas pd dataframe
- data.rename(columns={"data_name":"new_data_name"})
- data.info()
- data.isna().sum() # Get number of missing values in each column of the dataframe
- data.isna().sum().sum() # Get total number of rows that contain missing values
- (data.isna().sum().sum() / data.shape[0]) * 100 # Percentage of rows with missing values
- data.dropna() # Drop rows containing missing values 
- data.duplicated() # Investigate for duplicates
- data.duplicated().sum() # Investigate for duplicates
- data.head() # Display first few rows of dataframe
- data.shape # Get the shape of the dataframe (number of rows, number of columns)

- data.groupby("values").count().head() # Group by "director" and get the first few rows 
- data["Year"] # access the "Year" column
- data["Year"].sort_values() ## Sort values in "Year"
- data["Class"].value_counts(normalize=True)
- data.sort_values("Count", ascending = False) ## sort the rows of the dataframe in descending order by the "Count" column
- data.iloc[0:5] # get the first five rows of the sorted dataframe
- data["Year"] == 2025 # compare each year with 2025
- data.loc[data["Year"] == 2025, :] # for the rows where the year is 2025, access all the columns

- data["added"].dtype ## Get the data type of "added" column
- type(data["added"].iloc[0]) ## Get the data type of first entry in "added" column
- data["added"] = pd.to_datetime(data["added"]) ## Convert the "added" column to datetime format
- data["Year"].dtype # Get the data type of the "Year" column

- data.plot.barh(x = "Year", y = "Count") ## create a horizontal bar plot that displays the counts of your name over the years
- sns.countplot(x = "", data = )  # create a count plot that displays the count 
- sns.barplot(x = "", y = "", data = ) # create a bar plot
- sns.barplot(x = "", y = "", data = , ci = False) ## create a bar plot and hide the ci
- sns.histplot(data = , x = "", hue = "")
- sns.histplot(data)
- (np.percentile(data, 2.5), np.percentile(data, 97.5)) #between upper and lower limits of sd
- plt.legend(["name1", "name2"])
- plt.show() # display the plot on the screen
- sns.pairplot(data)

- plt.hist(listings["values"]) # create a histogram
- plt.hist(listings["values"], bins = np.arange(0, 1100, 40)) #specify the bins parameter 40
- plt.scatter(x = data["values1"], y = data["values2"]) # create a scatter plot to compare values
- plt.scatter(x = data["values1"], y = data["values2"], s = 5) # specify the s parameter (which represents the size of the points)
- plt.xlim(0, 1100) # set limits for the x-axis 
- plt.xlabel("x-axis name") # specify the x-axis label
- plt.ylabel("y-axis name") # specify the y-axis label
- plt.axhline(0)
- sm.qqplot(residuals, line='s')

- np.mean(data["value"])
- np.count_nonzero
- knn = KNeighborsClassifier(n_neighbors=3)
- knn.predict(X_train)
- knn.predict(y_train)
- knn.score(X_train, y_train)
- data.sample(frac = 0.75, random_state=0)
- data.drop(train.index)
- ols(formula = "value1 ~ value2", data = train ) #OLS Regression Results
- data.fit() #Fit model
- model.summary()
- model.resid

### webs

[# https://www.kaggle.com/](# https://www.kaggle.com/)

## 2 Statistic Basic

| **Term**           | **Definition**                                                                          |
|--------------------|-----------------------------------------------------------------------------------------|
| mean               | The sum of all the values in a data set divided by the number of values in the data set |
| median             | The middle value, in position, of an ordered data set                                   |
| mode               | The most frequently occurring value in a data set                                       |
| range              | The largest value minus the smallest value in a data set                                |
| standard deviation | A measure of variance in a data set                                                     |

## 3 Statistic Foundations

| **Term**              | **Definition**                                                                                                                                                                           |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| central limit theorem | A statistical theory stating that the distribution of sample means approaches a normal distribution as the sample size becomes larger, regardless of the population’s distribution       |
| confidence intervals  | A range of values derived from sample statistics that is likely to contain the value of an unknown population parameter, expressed at a specified confidence level (e.g., 95%)           |
| hypothesis            | A proposed explanation or prediction that can be tested through study and experimentation, often formulated as a null hypothesis (no effect) and an alternative hypothesis (some effect) |
| one-tailed test       | A type of hypothesis test where the area of interest is only in one tail of the distribution, used when testing for the possibility of the relationship in one direction                 |
| standard error        | The standard deviation of the sampling distribution of a statistic, typically the mean, indicating the precision of the sample mean estimate of the population mean                      |
| two-tailed test       | A type of hypothesis test where the areas of interest are in both tails of the distribution, used when testing for the possibility of the relationship in both directions                |
| type one error        | The error of rejecting a true null hypothesis (a false positive), denoted by alpha (α), often set at a significance level of 0.05                                                        |
| type two error        | The error of failing to reject a false null hypothesis (a false negative), denoted by beta (β), indicating a lack of power in the test                                                   |

## 4 Visual Studio Code .py

## import necessary Python libraries
import math
/
import os
from os import path
/
import datetime
from datetime import date, time, timedelta
import time
/
import shutil
from shutil import make_archive
from zipfile import ZipFile

### Basic Commands 
#### 1 hello world!, loops and functions 

*Basic data types in Python: Numbers, Strings, Booleans, Sequences, Dictionaries.*
*ERROR: variables of different types cannot be combined*

- mylist = [0, 1, "two", 3.2, False]
- mydict = {"one" : 1, "two" : 2}

- *print*(mylist) 
- *print*(mylist[2]) # to access a member of a sequence type, use []
- *print*(mylist[1:4:2]) # use slices to get parts of a sequence
- *print*(mylist[::-1]) # you can use slices to reverse a sequence
- *print*(mydict["two"]) # dictionaries are accessed via keys

- *del* mylist
- *del* mydict
- math.sqrt()
- math.pi

- datetime.datetime.now()

* Define a basic function*

def func1():
    print("I am a function")

def main():
    print("hello world!")
if __name__ == "__main__":
    main()

- conditional flow uses *if*, *elif*, *else*
- the match-*case* construct can be used for multiple comparisons

*loop*
- define a for loop

    for x in range(1,10):
        print (x)

- define a while loop

    while (x < 5):
        print(x)
        x = x + 1

- enumerate() #index the days

    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for i, d in enumerate(days):
        print (i, d)

#### 2 Files 
- file = open("file.txt","w+") #  writing and create it if it doesn't exist
- file = open("file.txt","a+") # appending text to the end
- file = open("file.txt","r") # read the contents

- * write ten new lines of data*
for i in range(10):   
      file.write("This is new line\n") 

- file.close() # close the file when done
- file.readlines() # reads the entire contents of a file line by line into an array

- path.exists("file.txt")
- path.isfile("file.txt")
- os.rename("file.txt", "newfile.txt")

#### 3 dates and times

#### 4 Internet data




### webs

[https://www.kaggle.com/](https://www.kaggle.com/)