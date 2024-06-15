import numpy as np
from numpy import array
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers.merge import concatenate

# KERAS INTRO FOR DEEP LEARNING
# https://machinelearningmastery.com/keras-functional-api-deep-learning/


# MLP multivariate second approach 
# Each Input series can be handled by a separate MLP and the output of each of these submodels
# can be combined before a predicition is made for the output sequence 
# It may offer more flexibility or better performance depending on the specifics of the problem 
# that are being modeled.

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
for i in range(len(X)):
	print(X[i], y[i])
#separate input data
#catch column and traspose again to column array
X1 = X[:,:,0]
X2 = X[:,:,1]
for i in range(len(X1)):
	print(X1[i], X2[i],y[i])
# first input model 
visible1 = Input(shape=(n_steps,))
dense1 = Dense(100,activation='relu')(visible1)
# second input model 
visible2 = Input(shape=(n_steps,))
dense2 = Dense(100,activation='relu')(visible2)
# merging input models 
merge = concatenate([dense1,dense2])
output = Dense(1)(merge)
model = Model(inputs=[visible1,visible2],outputs=output)
model.compile(optimizer='adam', loss='mse')
#fit model
model.fit([X1,X2],y,epochs=2000,verbose=0)
train_loss = model.history.history['loss']
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps))
x2 = x_input[:, 1].reshape((1, n_steps))
yhat = model.predict([x1, x2], verbose=0)
print(yhat)
