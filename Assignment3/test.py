import numpy as np 

def create_tree (tree):
	tree = [1,2]
	tree[0].append(['a'])
	tree[1].append(['n'])
	# tree[0].append(['a','b'])
tree = []
tree = create_tree(tree)
print (tree)