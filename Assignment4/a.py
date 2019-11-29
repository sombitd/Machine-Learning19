import numpy as np 
import math
import pandas as pd
import random
filename = "data4_19.csv"
k = 3
no_iterations = 10
def get_data_train(filename):

    train_data = []  #to store training data from csv file
    data = []

    #read data from csv
    with open(filename,"r") as csv_data:
        data = csv_data.read().split('\n')
    # print (len(data))
    for i in range(len(data)-1):    
        row = data[i].split(',')

        train_data.append(row)

    df = pd.DataFrame(train_data,columns=['label1','label2','label3','label4','targets'])   
    return df,data
labels = ['label1','label2', 'label3' , 'label4' ]

def get_label(inst,mean):
    dis = np.zeros(k)
    for k_i in range(k):
        for y in range(len(labels)):
            dis[k_i] += pow((mean[k_i][y]-float(inst[labels[y]])),2)
    ans = np.argmin(dis)
    return ans

def iterate():
    df,data = get_data_train(filename)
    index = np.zeros(df.shape[0],dtype = int)
    mean =  np.random.rand(k,len(labels))
    for k_i in range(k): 
        # n = random.randint(0,df.shape[0])
        n =55*k_i+30                           ## To initialize the data point in the different gound truth labels to ensure proper k-means clustering 
        for i in range (len(labels)):
            # n = int(n)
            mean[k_i][i] = df.iloc[n][labels[i]]
    print("Random Mean ",mean)
    print("#######################################")
    target =df[df.columns[-1]]
    target = target.unique()
    # print(target.shape)
    for _ in range(no_iterations):
        for i in range(df.shape[0]):
            index[i] =get_label(df.iloc[i],mean)
        for k_i in range(k):
            temp = np.where(index == k_i)
            temp = np.asarray(temp)
            temp = temp[0]
            for y in range(len(labels)):
                val =[]
                for num in temp:
                    val.append( float(df.iloc[num][labels[y]]) )

                if( len(val)):
                    mean[k_i][y] = sum(val)/len(val)    
        
    print("new mean after", no_iterations,"iterations ", mean)
    print("######################")   
    jac = np.zeros((k,len(target)))
    for i in range(df.shape[0]):
        index[i] =get_label(df.iloc[i],mean)
    jac = np.zeros((k,len(target)))
    for k_i in range(k):
        j=0
        for var in target:
            pos =0      ## intersection
            total = 0   ## union
            for i in range (df.shape[0]):
                # print(index[i])
                if(df.iloc[i]['targets'] == var and target[index[i]] == var and index[i] == k_i ):
                    pos = pos+1
                if(df.iloc[i]['targets'] == var or target[index[i]] == var):
                    total = total+1
            jac[k_i][j] = 1- pos/total
            j = j+1
    print("Jaccard Matrix")
    print(jac)
    print("######################")       
    for k_i in range(k):
        id = np.argmin(jac[k_i])
        print("Cluster ", k_i ," resembles " , target[id], "with Jaccard Distace " , jac[k_i][id])
iterate()