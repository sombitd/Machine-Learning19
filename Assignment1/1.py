'''           
Name : Sombit Dey 
Roll No. 17EC10056  

to excecute : change the file path, 
            python3.5 17EC10056_1.py

'''


import numpy as np 
import math
label1 = ["1st","2nd","3rd","crew"]
label2 = ['adult','child']
label3 = ['male','female']
target_label = ['yes','no']
complete = [label1, label2 , label3 ,target_label]


filename = '/home/sombit/Machine-Learning19/Assignment1/data1_19.csv'
def get_data():
   
    train_data = []  #to store training data from csv file
    data = []

    #read data from csv
    with open(filename,"r") as csv_data:
        data = csv_data.read().split('\n')

    for i in range(len(data)-1):    
        row = data[i].split(',')
        train_data.append(row)
    return np.asarray(train_data)   

def entropy( data,label,naya):
    ans = np.zeros((len(naya[label]),2))
    for i in range(len (data)):
        for k in range ( len ( naya[label])):
            # print("data[i][label]",data[i][label] , "  complete[label][k]" , complete[label][k])
            if(data[i][label]== naya[label][k]):
                if(data[i][-1] == target_label[0]):
                    ans[k][0]+=1
                else:
                    ans[k][1]+=1
                break
    val = 0.0
    total = len (data)
    # print (total)
    for k in range ( len ( ans)):
        sub_sum = ans[k][0] + ans[k][1]
        # print ( "sub_sum",sub_sum)
        if(ans[k][0]==0):
            val = val  
        elif(ans[k][1]==0):
            val = val  
        else:
            val = val  + (-ans[k][0]*math.log((float(ans[k][0])/float(sub_sum)) ,2)  -ans[k][1]*math.log(float(ans[k][1])/float(sub_sum),2))/float(total)   
    return val

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
def get_m(data):
    pos = 0
    neg = 0
    for i in range( len (data)):
        if(data[i][-1] == complete[-1][0]):
            pos +=1
        else:
            neg+=1
    if(pos>neg ):
        return 1
    return 0
def create_tree(data,level,compl):
    if(len(data[0]) <1):
        return
    first = entropy_initial(data)
    if (first ==0 or len(data[0])==1) :
        for j in range(level):
            print("--",end=" ")
        if(get_m(data)==1):
            print(" YES")
        else:
            print (" NO")
        return 

    # print (entropy( data ,2))
    max_label = 0
    max_sum = first - entropy(data,0,compl) 

    for k in range (1, len(data[0])-1):
        curr_sum = entropy(data,k,compl);
        if( (first - curr_sum ) > max_sum):
            max_sum = first - curr_sum
            max_label = k
    if( max_sum == 0):
        for j in range(level):
            print("--",end=" ")
        if(get_m(data)==1):
            print(" YES")
        else:
            print (" NO")
        return 

    for k in range (len(compl[max_label])):
        data_new=[]

        for j in range(level):
            print("--",end=" ")
        print (compl[max_label][k])
        for i in range(len(data)):
            if(data[i][max_label] == compl[max_label][k]):
                data_new.append(data[i])
        data_new=np.delete(data_new,max_label,axis=1)
        # print(data_new.shape)
        naya = compl.copy()
        del naya[max_label]

        create_tree( data_new , level+1,naya)
    return



def main ():
    data = get_data()
    data = np.delete(data,0,0)
    # first = entropy_initial(data)
    # # print (entropy( data ,2))
    # max_label = 0
    # max_sum = first - entropy(data,0) 
    # for k in range (1, len(data[0])-1):
    #     curr_sum = entropy(data,k);
    #     if( (first - curr_sum ) > max_sum):
    #         max_sum = first - curr_sum
    #         max_label = k
    create_tree(data , 0 , complete)
    

            

main()
