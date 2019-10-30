'''           
Name : Sombit Dey 
Roll No. 17EC10056  

to excecute : change the filename , 
            python3.5 17EC10056_a2.py


'''

import numpy as np
import pandas as pd

filename_train  = '/home/sombit/Machine-Learning19/Assignment2/data2_19.csv'
filename_test  = '/home/sombit/Machine-Learning19/Assignment2/test2_19.csv'

def prepare_dataframe(df):    # create data-frame
    temp = []
    for j in range(len(df)):
        string = ""
        for i in range(len(df.iloc[j,:])):
            string += str(df.iloc[j,i])
        temp.append([int(char) for char in string.split(',')])
    df = pd.DataFrame(temp, columns = df.columns.values[0].split(","))
    return df 
def pos_prob(data):     # get the count of positive and negative numbers
    pos = 0
    neg = 0
    for i in range (len(data)):
        if(data[i][0]==1) :
            pos = pos+1
        else : 
            neg = neg+1
    # print ("pos",pos," neg ",neg)
    return pos,neg
def temp(data):          ### build the probabilty table
    table_a = np.zeros((6,5))
    table_b = np.zeros((6,5))
    
    for i in range(len(data)):
        if(data[i][0]==1):
            for j in range(len(data[0])-1):
                # print ("j",j)
                table_a[j][data[i][j+1]-1] =table_a[j][data[i][j+1]-1]+1
        else:
            for j in range(len(data[0])-1):
                # print ("j",j)
                table_b[j][data[i][j+1]-1] =table_b[j][data[i][j+1]-1]+1 
    return table_a,table_b

def main ():
    data = pd.read_csv(filename_train) 
    data = prepare_dataframe(data)
    data =pd.DataFrame(data).to_numpy()
    test = pd.read_csv(filename_test) 
    test = prepare_dataframe(test)
    test =pd.DataFrame(test).to_numpy()
    # data = data[:10]
    # print(data)
    pos,neg = pos_prob(data)
    # print ( "pos ",pos, " ",neg )
    table_a,table_b = temp(data)             ### table_a contains the positive probabilty and table_b contains the negative probabilty
    predict = []
    for i in range (len(test)):
        val1 = pos
        val0 = neg
        for  j in range (len(test[0])-1 ):
            val1 *= (table_a[j][test[i][j+1]-1] +1)/(pos+6)
            val0 *= (table_b[j][test[i][j+1]-1] +1)/(neg+6)
        # val1 = val1 * pos
        # val0 = val0 * neg
        if(val1 > val0 ) :
            predict.append(1)
            # print("true ")
        else: 
            predict.append(0)
            # print ("false")
    correct =0.0
    total = 0
    for i in range (len(predict) ):
        total = total +1
        if(predict[i]==test[i][0]):
            correct = correct+1
    print(correct/total)        # cout << 0 << endl;


main()