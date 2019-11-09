'''           
Name : Sombit Dey 
Roll No. 17EC10056  

to excecute : 
            python3.5 17EC10056_3.py


'''

import numpy as np 
import pandas as pd
import math

label1 = ["1st","2nd","3rd","crew"]
label2 = ['adult','child']
label3 = ['male','female']
target_label = ['yes','no']
features = ['pclass','age','gender']
complete = [label1, label2 , label3 ,target_label]
filename1 = 'data3_19.csv'
filename2 = 'test3_19.csv'
no_iteration = 3
def get_data_train(filename):

    train_data = []  #to store training data from csv file
    data = []

    #read data from csv
    with open(filename,"r") as csv_data:
        data = csv_data.read().split('\n')

    for i in range(len(data)-1):    
        row = data[i].split(',')
        # for i in range(len(row)):
        #   row[i] = int(row[i])
        train_data.append(row)
    data = train_data[1:]
    df = pd.DataFrame(data,columns=['pclass','age','gender','survived'])   
    return df,data

def get_data_test(filename):
    train_data = []  #to store training data from csv file
    data = []

    #read data from csv
    with open(filename,"r") as csv_data:
        data = csv_data.read().split('\n')

    for i in range(len(data)-1):    
        row = data[i].split(',')
        train_data.append(row)
    # data = train_data[]
    df = pd.DataFrame(train_data,columns=['pclass','age','gender','survived'])   
    return df

def entropy(data,attribute,prob):   ## entropy for split
    Class = data.keys()[-1]   
    target = data[Class].unique()  
    total_var = data[attribute].unique()    
    total_entropy = 0
    for var in total_var:
        entropy = 0
        for var2 in target:
            num = data[attribute][data[attribute]==var][data[Class] ==var2]
            # print("test ", num.index)
            # num = data.index.get_loc(data.index[data[attribute][data[attribute]==variable][data[Class] ==target_variable]][0])
        
            x  = [prob[i] for i in num.index]
            x = sum(x)
            num = len(num)

            total_part = len(data[attribute][data[attribute]==var])
            # fraction = num/(den)
            # indices = [i  for i in range(data.shape[0]) if(data.iloc[i][attribute][data.iloc[i][attribute]==variable][data.iloc[i][Class] ==target_variable])] 
            # temp = data[attribute][data[attribute]==variable]
            # print(temp.shape[0])
            # for i in range(temp.shape[0]):
            #     if(temp.iloc[i][temp.iloc[i][Class] ==target_variable]):
                    # fraction+= pro
            # entropy += -fraction*math.log(fraction+0.0001,2)
            entropy += -x*math.log(x+0.0001,2)
        part = total_part/len(data)
        total_entropy += -part*entropy
    return abs(entropy)

def entropy_initial(data,prob):   ## initial entropy
    pos = 0
    neg = 0
    for i in range( len(data) ):
        if(data[i][-1] == complete[-1][0]):
            pos += prob[i]
        else:   
            neg+= prob[i]

    if(neg==0 or pos ==0):
       val  = 0

    else:
        val  = -pos*math.log(float(pos),2)- neg
    return val


def create_tree(df,data,features,prob):
    target = df.keys()[-1]                
    number = df[target].unique()
    if (len(number) ==1):
        return number[0]
    if(len(features) == 0):
        arr = df[target]
        arr= arr.value_counts()
        return arr.idxmax()     
    y =[]    
    for x in features:
        y.append(entropy_initial(data,prob)-entropy(df,x,prob))
    label = features[np.argmax(y)]
    max_label = np.argmax(y)
    # print ( max_label)   
    tree = {label: {}}
    features_n = []
    for x in features:
        if(x != label):
            features_n.append(x)
    features_rem = df[label].unique()
    data_new = []
    for x in features_rem:
        for k in range (df.shape[0]):
            if(df.get_value(k ,label) == x):
                data_new.append(data[k])
        new_df= df[df[label] == x].reset_index(drop=True)        
  
        tree[label][x] = create_tree( new_df, data_new , features_n,prob)
            
    return tree

def predict(example,tree):

    for x in tree.keys():
        # print("x",x)
        test = example[x]
        tree = tree[x][test]
        ans =0
        if type(tree) is dict:
            ans = predict(example,tree)
        else:
            ans = tree
            break;  
    return ans
# print(df.iloc[0])
# print(df.iloc[0]['pclass'])
# # print(predict(df.iloc[0],tree))
# print(df.iloc[0][-1])
def acc(df,tree):
    ans  = []
    for x in range (len(df)):
        if(predict(df.iloc[x],tree) == df.iloc[x][-1]): 
            ans.append(1)   
        else:
            ans.append(0)
    return ans

def acc_test(df,tree):
    ans  = []
    for x in range (len(df)):
        if(predict(df.iloc[x],tree) == df.iloc[x][-1]): 
            ans.append(1)   
        else:
            ans.append(0)

    print("accuracy = " , sum(ans)/len(ans) )
    # print(ans)
    return ans

def ensemble():
    prob = np.ones(df.shape[0])/df.shape[0] 
    for _ in range(no_iteration):
        # prob = np.ones(df.shape[0])/df.shape[0] 
        tree = create_tree(df,data,features,prob)
        ans = acc(df,tree )
        error = 1- sum(ans)/len(ans)
        fac = 0.5 * math.log((1-error)/error)
        for i in range (len(ans)):
            if(ans ==0 ):
                prob[i] = prob[i]*math.exp(fac)
            else :
                prob[i] = prob[i]*math.exp(-fac)
        prob = prob/sum(prob)
    acc_test(df_test,tree)

df, data = get_data_train(filename1)
df_test = get_data_test(filename2)
ensemble()


