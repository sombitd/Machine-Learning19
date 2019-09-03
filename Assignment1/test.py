import math
import numpy as np

def dataRead(filename):
	"""Read data from csv file"""
	data = np.genfromtxt(filename, delimiter = ',')	
	return data 


class Node:
	def __init__(self, data):
		self.left = None
		self.right = None
		self.data = data


def split(X, Y, gainMax_attr):
	""" Splits the dataset """
	X_1 = []
	X_0 = []
	Y_1 = []
	Y_0 = []
	for i in range(0, X.shape[0]):
		if(X[i][gainMax_attr] == 0):
			X_0.append(X[i])
			Y_0.append(Y[i])
		else:
			X_1.append(X[i])
			Y_1.append(Y[i])
	X_0 = np.reshape(np.array(X_0), (-1, X.shape[1]))
	X_1 = np.reshape(np.array(X_1), (-1, X.shape[1]))
	Y_0 = np.array(Y_0)
	Y_1 = np.array(Y_1)
	return X_0, X_1, Y_0, Y_1


def entropy(p, n):
	""" Calculates entropy"""
	if(p == 0 and n == 0):
		ent = 0
	elif(p == 0):
		ent = -(n/(p + n))*math.log((n/(p + n)), 2)
	elif(n == 0):
		ent = -(p/(p + n))*math.log((p/(p + n)), 2)
	else:
		ent = -(p/(p + n))*math.log((p/(p + n)), 2) -(n/(p + n))*math.log((n/(p + n)), 2)
	return ent



def buildTree(X, Y, marked):
	if(X.shape[1] == len(marked)):
		return None
	flag_neg = flag_pos = 1
	for i in range(0, Y.shape[0]):
		if(Y[i] == 1):
			flag_neg = 0
		else:
			flag_pos = 0

	if(flag_neg):
		return Node(-1)
	if(flag_pos):
		return Node(-2)

	E_class = 0.0
	p = n = 0.0
	for i in range(0, X.shape[0]):
		if(Y[i] == 1):
			p += 1
		else:
			n += 1
	E_class = entropy(p, n)

	gainMax = -1
	gainMax_attr = -1
	for i in range(0, X.shape[1]):
		if(i in marked):
			continue
		p1 = n1 = p2 = n2 = pos = neg = 0.0
		for j in range(0, X.shape[0]):
			if(X[j][i] == 0):
				neg += 1
				if(Y[j] == 1):
					p1 += 1
				else:
					n1 += 1
			else:
				pos += 1
				if(Y[j] == 1):
					p2 += 1
				else:
					n2 += 1

		I_G = (pos/(pos + neg))*entropy(p2, n2) + (neg/(pos + neg))*entropy(p1, n1)
		gain = E_class - I_G
		if(gainMax < gain):
			gainMax = gain
			gainMax_attr = i

	marked.append(gainMax_attr)
	root = Node(gainMax_attr)

	X_0, X_1, Y_0, Y_1 = split(X, Y, gainMax_attr)
	root.left = buildTree(X_0, Y_0, marked)
	root.right = buildTree(X_1, Y_1, marked)

	return root


def predict(tree, test_data):
	ans = []
	for i in range(0, test_data.shape[0]):
		temp = tree
		while(temp):
			if(temp.data == -1):
				ans.append(0)
				break
			if(temp.data == -2):
				ans.append(1)
				break
			temp = temp.left if test_data[i][temp.data] == 0 else temp.right
	output(ans)


def output(ans):
	file = open('16ME30035_2.out', 'w')
	for i in ans:
		file.write(str(i) + ' ')
	file.close()
	print ans[0],ans[1], ans[2], ans[3]


def main():
	train_data = dataRead('data2.csv')
	X = train_data[:, 0 : train_data.shape[1] - 1]
	Y = train_data[:, train_data.shape[1] - 1]
	n_features = X.shape[1]
	mark = np.zeros([1, n_features], dtype = np.int32)
	decision_tree = buildTree(X, Y, [])

	testData = dataRead('test2.csv')
	predict(decision_tree, testData)
	 

main()