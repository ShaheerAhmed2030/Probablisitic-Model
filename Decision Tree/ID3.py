# Importing Libraries
import pandas as pd
import numpy as np
from pprint import pprint

# Functions  
# Entropy Function
def Entropy(Target):
    Elements,Count = np.unique(Target,return_counts = True)
    entropy = np.sum([(-Count[i]/np.sum(Count))*np.log2(Count[i]/np.sum(Count)) for i in range(len(Elements))])
    return entropy

# Information Gain
def InformationGain(data,Branch,target="COVID-19"):
    total_entropy = Entropy(data[target])
    Vals,Count = np.unique(data[Branch],return_counts = True)
    Average = np.sum([(Count[i]/np.sum(Count))*Entropy(data.where(data[Branch]==Vals[i]).dropna()[target]) for i in range(len(Vals))])
    return total_entropy - Average

# Algorithm
def ID3(data,originalData,features,target_attribute_name="COVID-19",parent_node_class=None):
    # if all values are 0 or all 1
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    #if the dataset is empty
    elif len(data) == 0:
        return np.unique(originalData[target_attribute_name])[np.argmax(np.unique(originalData[target_attribute_name],return_counts=True)[1])]
        
    # no features
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts= True)[1])]

    # Calculating the column with maximum gain
    Items = [InformationGain(data, i ,target_attribute_name) for i in features]
    Highest = np.argmax(Items)
    Feature_to_pick = features[Highest]
     
    # Removing the feature which we picked from the list
    
    features = [i for i in features if i!= Feature_to_pick]
    # Creating the Tree
    Tree = {Feature_to_pick:{}}
    # Using recursion to grow the tree further
    for i in np.unique(data[Feature_to_pick]):
        sub_data = data.where(data[Feature_to_pick] == i).dropna()
        Sub_Tree = ID3(sub_data,originalData,features,target_attribute_name,parent_node_class)
        # Inserting the tree
        Tree[Feature_to_pick][i] = Sub_Tree
    return Tree

# Prediction by iterating the tree
def Prediction(query,tree,default = 1):
    for i in list(query.keys()):
        if i in list(tree.keys()):
            try:
                result = tree[i][query[i]]
            except:
                return default
            
            result = tree[i][query[i]]
            if isinstance(result,dict):
                return Prediction(query, result)
            else:
                return result
     
# Splitting Dataset
def Training(Data,Spliter):
    training_Data = Data.iloc[:Spliter].reset_index(drop = True)
    testing_Data = Data.iloc[Spliter:].reset_index(drop = True)
    return training_Data,testing_Data


# Algorithm Testing
def Test(data,tree,target = "COVID-19"):
    Queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns = ["predicted"])
    
    #Accuracy
    for i in range(len(data)):
       predicted.loc[i,"predicted"] = Prediction(Queries[i],tree,1.0)
    print("Algorithm Prediction Accuracy:",(np.sum(predicted["predicted"]==data[target])/len(data))*100,'%')
    
# Query Prediction
def TestPositive(data,tree):
    Queries = data.iloc[:,:,].to_dict(orient = "records")
    print("Test Results:",end=" ")
    if (Prediction(Queries[0],tree,1.0) == 1):
        print("Corona Positive")
    else:
        print("Corona Negative")


# Main Method
def Main():
    # Importing Dataset
    Data = pd.read_csv("Cleaned.csv")
    Data.drop(Data.columns[0],axis = 1, inplace = True)
    # Splitting up the data
    Training_data , Testing_Data = Training(Data,400)
    Query = pd.read_csv("Query.csv")
    tree = ID3(Training_data,Training_data,Training_data.columns[:-1])
    pprint(tree)
    Test(Testing_Data,tree)
    TestPositive(Query, tree)   

Main()