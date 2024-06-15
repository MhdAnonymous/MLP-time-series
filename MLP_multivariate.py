import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense


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
# transpose arrays
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
#hstack => stack arrays in sequence horizontally (column wise)
dataset = np.hstack((in_seq1, in_seq2, out_seq))
print(dataset)
#Split the data into samples maintaining the order of observations across the two input sequences
#That is, the first three time steps of each parallel series are provided as input to the model and 
# the model associates this with the value in the output series at the third time step
n_steps = 3
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])
#MLPs require that the shape of the input portion of each sample is a vector. With a multivariate 
# input, we will have multiple vectors, one for each time step.
# the length of each input vector is the number of time steps multiplied by 
# the number of features or time series. 
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
for i in range(len(X)):
	print(X[i], y[i])
# define MLP model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)
