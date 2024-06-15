import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
#https: // machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)



if __name__ == "__main__":
	x = [10,20,30,40,50,60,70,80,90]
	X,Y = split_sequence(x,3)
	# Define Model
	model = Sequential()
	model.add(Dense(100,activation='relu', input_dim = 3))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X,Y, epochs=2000, verbose=0)
	#Demonstrate prediction
	x_input = np.array([70,80,90])
	x_input = x_input.reshape((1,3))
	yhat = model.predict(x_input,verbose=0)
	print(yhat)
