'''           
Name : Rishabh Singh 	Roll No. 17QE30004
Machine Learning Assignment 1
to excecute :
If the data file is in some other directorty,
change the filename,
Dependency Libraries : numpy, pandas, math
On Terminal :    python3.5 17QE30004_1.py
'''

import numpy as np 
import math
import pandas as pd

data_file = "data3_19.csv" #importing data

def getData():
    read_data = []  #to store training data from csv file
    data = []

    #read data from csv
    with open(data_file, 'r') as file: #reading data
    	read_data = file.read().split('\n') #spliting by lines

    data = read_data[:-1] #saving as list
    #print(data
    extract = [[],[],[],[]] 
    data_dict = {} #dictionary 
    data = [vals.split(',') for vals in data] #spliting by commas
    features = data[0] #extracting features : pclass , age, gender, survival
    #print(features)

    data = data[1:] # saving data from 1st row, removing features column
    #print(data)
    
    for i in range(0, len(features)): 
        for r in data:
            extract[i].append(r[i])

    for idx,val in enumerate(features):
        data_dict[val] = extract[idx]
    
    df = pd.DataFrame(data_dict,columns=['pclass','age','gender','survived'])
    #print(df['survived'].value_counts()['yes'])
    
    return df , features

data , att1 = getData()
print (data)
def findEntropy(data):
    Class = data.keys()[-1]   
    entropy = 0
    values = data[Class].unique()
    for value in values:
        fraction = data[Class].value_counts()[value]/len(data[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

entropy = findEntropy(data)

def findEntropyAtt(data,attribute):
    Class = data.keys()[-1]   
    target_variables = data[Class].unique()  
    variables = data[attribute].unique()    
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(data[attribute][data[attribute]==variable][data[Class] ==target_variable])
            den = len(data[attribute][data[attribute]==variable])
            fraction = num/(den+0.0001)
            entropy += -fraction*math.log(fraction+0.0001,2)
        fraction2 = den/len(data)
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def findWinner(data, att):
    Entropy_att = []
    root = []
    for key in att:
        root.append(findEntropy(data)-findEntropyAtt(data,key))
    return att[np.argmax(root)]

def get_subtable(data, node,value):
    return data[data[node] == value].reset_index(drop=True)

att = ['pclass','age','gender']

def buildTree(data,att): 
    Class = data.keys()[-1]   
    val = data[Class].unique() 
    if len(val) == 1:
        return val[0]
    if(len(att) == 0):
        return data[Class].value_counts().idxmax()
    node = findWinner(data,att)
    tree_node = {node: {}}
    att_new = list()
    for val in att:
        if(val != node):
            att_new.append(val)
   
    attValue = np.unique(data[node])
    
    for value in attValue:
        subtable = get_subtable(data,node,value)                        
        subtree = buildTree(subtable, att_new)
        tree_node[node][value] = subtree
    return tree_node
  
print(buildTree(data,att))   