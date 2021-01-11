# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Loading dataset
Dataset = pd.read_csv("Covid Dataset.csv")
# Cleaning
# Check if they are any columns which contains NAN
print(Dataset.isna().any())
# Dropping Columns which are not related to our problem
Dataset.drop(["ID"],axis = 1,inplace = True)
Dataset.drop(["Country"],axis = 1,inplace = True)
Dataset.drop(["Sanitization from Market"],axis = 1,inplace = True)
Dataset.drop(["Wearing Masks"],axis = 1,inplace = True)
Dataset.drop(["Family working in Public Exposed Places"],axis = 1,inplace = True)
Dataset.drop(["Visited Public Exposed Places"],axis = 1,inplace = True)
Dataset.drop(["Attended Large Gathering"],axis = 1,inplace = True)
Dataset.drop(["Gastrointestinal"],axis = 1,inplace = True)
Dataset.drop(["Hyper Tension"],axis = 1,inplace = True)
Dataset.drop(["Diabetes"],axis = 1,inplace = True)
Dataset.drop(["Running 0se"],axis = 1,inplace = True)
Dataset.drop(["Chronic Lung Disease"],axis = 1,inplace = True)
Dataset.drop(["Heart Disease"],axis = 1,inplace = True)
Dataset.drop(["Asthma"],axis = 1,inplace = True)
CleanedDataset = Dataset.sample(frac = 0.1)
CleanedDataset.to_csv("Cleaned.csv")
rows,cols = CleanedDataset.shape
print("Rows:",rows)
print("Columns:",cols)        

Data = CleanedDataset.drop(columns = ['COVID-19'])
# Plotting 
# Data Visualization
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, CleanedDataset.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(CleanedDataset.columns.values[i - 1])
    vals = np.size(CleanedDataset.iloc[:, i - 1].unique())  
    plt.hist(CleanedDataset.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Plotiing Pie charts
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, CleanedDataset.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(CleanedDataset.columns.values[i - 1])
    values = CleanedDataset.iloc[:, i - 1].value_counts(normalize = True).values
    index = CleanedDataset.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')