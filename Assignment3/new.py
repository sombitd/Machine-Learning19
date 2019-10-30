import numpy as np 
import pandas as pd
import math

label1 = ["1st","2nd","3rd","crew"]
label2 = ['adult','child']
label3 = ['male','female']
target_label = ['yes','no']
complete = [label1, label2 , label3 ,target_label]
filename1 = '/home/sombit/Machine-Learning19/Assignment3/data3_19.csv'
filename2 = '/home/sombit/Machine-Learning19/Assignment3/test3_19.csv'
def get_data(filename):
    # my_data = np.genfromtxt('/home/sombit/Machine-Learning19/Assignment1/data1_19.csv', delimiter=',')
    # return my_data
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

def entropy(data,attribute):
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
def entropy_initial(data):
    pos = 0
    neg = 0
    for i in range( len(data) ):
        if(data[i][-1] == complete[-1][0]):
            pos +=1
        else:   
            neg+=1

    if(neg==0 or pos ==0):
       val  = 0

    else:
        val =  (-pos*math.log((float(pos)/float(pos+neg)) ,2)  -neg*math.log(float(neg)/float(pos+neg),2))/(pos+neg)   
    return val

features = ['pclass','age','gender']
def create_tree(df,data,features):
    target = df.keys()[-1]
    number = df[target].unique()
    if (len(number) ==1):
        return number[0]
    if(len(features) == 0):
        return df[target].value_counts().idxmax()
    y =[]    
    for x in features:
        y.append(entropy_initial(data)-entropy(df,x))
    label = features[np.argmax(y)]
    max_label = np.argmax(y)
    print ( max_label)   
    root = features[max_label]
    tree = {root: {}}
    features_n = []
    for x in features:
        if(x != root):
            features_n.append(x)
    features_rem = df[root].unique()
    data_new = []
    for x in features_rem:
        for k in range (df.shape[0]):
            # print (df[k][root])
            if(df.get_value(k ,root) == x):
                # new_df.append(df[k])        
                data_new.append(data[k])
        new_df= df[df[root] == x].reset_index(drop=True)        
        # new_df.reset_index()
        # data_new = data[df[root]==x]
        tree[root][x] = create_tree( new_df, data_new , features_n)
            
    return tree
# def predict(df,tree):
#     while(tree ):
#         get_value(x,root)
#         tree = tree[]

df, data = get_data(filename1)
df_test,data_test = get_data(filename2)

tree = (create_tree(df,data,features))
print (tree)
print (tree['gender'])
def predict(data)
x = tree.keys()
value = instance[x]
tree = tree[x][value]
if type(tree ) is dict :
    ans = predict(instance,tree)
else :
    ans = tree
    break;
return ans

def predict(example,tree):

    x = tree.keys()
    test = example[x]
    tree = tree[x][test]
    if type(tree) is dict:
        ans = predict(test,tree)
    else:
        ans = tree
        break;
    return ans

