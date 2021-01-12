# Importing libraries
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


# Splitting the dataset into test and training samples
def Train_Test_Split(Data,ratio):
    Test_DF = Data.sample(frac = ratio)
    Test_indeces = Test_DF.index.tolist()
    Train_DF = Data.drop(Test_indeces)
    return Train_DF,Test_DF


# Frequency Table (For training our algorithm)
def Freq_Table(Data,features,Target = "COVID-19"):
    table = {}
    counts = Data[Target].value_counts().sort_index()
    table["Attribute"] = counts.index.to_numpy()
    table["Count"] = counts.values
    # Now we will create probabilities of each feature
    for feature in features:
        table[feature] = {}
        counts = Data.groupby(Target)[feature].value_counts()
        df_counts = counts.unstack(Target)
        pprint(df_counts)
        print()
        # replace nan with 0
        if df_counts.isna().any(axis = None):
            df_counts.fillna(value = 0,inplace=True)
            df_counts += 1
        # Calculating probabilites and storing in Table     
        df_probabilities = df_counts/df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].to_numpy()
            table[feature][value] = probabilities
    return table


    
# Prediction by Accessing Frequency Table
def Prediction(Query,Table):
    # Total 0s and 1s outputs
    Predict = Table["Count"]
    # Calculating probability for a query
    for feature in Query.index:
        # For rare cases outside of our training dataset
        try:
            value = Query.loc[feature]
            probability = Table[feature][value]
            # P(B1|A)*P(B2|A) ....
            Predict = Predict*probability
        except KeyError:
            continue
    Highest = Predict.argmax()
    # Highest Probability will be the likelihood event
    prediction = Table["Attribute"][Highest]
    return prediction

# Algorithm Accuracy and testing
def Test(data,Table,target = "COVID-19"):
    data.reset_index(inplace = True)
    Queries = data.iloc[:,:-1]
    predicted = pd.DataFrame(columns = ["predicted"])
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = Prediction(Queries.loc[i],Table)
    # calculating Accuracy
    print("Algorithm Prediction Accuracy:",(np.sum(predicted["predicted"]==data[target])/len(data))*100,'%')


# Query testing
def TestPositive(Query,Table):
    print("Query Test Results:",end = " ")
    if (Prediction(Query,Table) == 1):
        print("Corona Positive")
    else:
        print("Corona Negative")


# Main Method    
def Main():
    # Importing Dataset
    Data = pd.read_csv("Cleaned.csv")
    Data.drop(Data.columns[0],axis = 1, inplace = True)
    # Reading Query to predict output
    Query = pd.read_csv("Query.csv").loc[0]
    # Getting features (Data attributes)
    Features = Data.columns[:-1]
    # 80% data for training, other 20 for testing        
    Train_DF,Test_DF = Train_Test_Split(Data, 0.30)
    # Creating frequency table
    print("Frequency Table:")
    Frequency_Table = Freq_Table(Train_DF, Features)
    print("Frequency Table (Probabilities):")
    pprint(Frequency_Table)     
    Test(Test_DF,Frequency_Table)
    TestPositive(Query, Frequency_Table)
    #Plot("Breathing Problem",Frequency_Table)
    
# Plotting Function
def Plot(Feature,Table):
    X = [0,1]
    Y = [Table[Feature][0][1],Table[Feature][1][1]]
    plt.bar(X,Y,width = 0.2)
    plt.xticks(X)
    plt.title(Feature)
    plt.xlabel("X (0 or 1)")
    plt.ylabel("P(COVID-19|" + str(Feature) +")")
    
Main()