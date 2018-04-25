from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
target = [(i+5)/100 for i in range(100)]

data = np.array(Data, dtype=float)
target = np.array(target, dtype = floa

x_train, x_test, y_train, y_test = train_test_split(Data, target, test_size=0.2, random_state=4)
#Neural Network
model = Sequential()
model.add(LSTM((1),batch_input_shape=(None,5,1), return_sequences=True))
model.add(LSTM((1), return_sequences=False))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
#Training The Model 
history = model.fit(np.array(x_train),np.array(y_train),epochs=400, validation_data=(np.array(x_test), np.array(y_test)))

#Result
result = model.predict(np.array(x_test))
#Graph Showing Prediction and Target
plt.scatter(range(20),result, c='r')
plt.scatter(range(20),y_test,c='b')
plt.show()