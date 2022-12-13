from numpy import loadtxt
from numpy import savetxt
import numpy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow
from keras.layers import LSTM, Activation, Bidirectional
from keras.layers import TimeDistributed

# setting the seed
seed(1)
set_seed(1)

index1 = 2409

# load the data from the csv file
orig_epochs = loadtxt('loaded_complete_data_2_sec.csv', delimiter=',')
print(orig_epochs.shape)

# shuffle the training data
numpy.random.seed(2) 
numpy.random.shuffle(orig_epochs)
print(orig_epochs.shape)

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(orig_epochs[0:index1, :], orig_epochs[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)


print("********************")


# augment train data
X_total = numpy.append(X_train_tmp, X_train_tmp, axis=0)
print(X_total.shape)
print(X_total[:, -1].astype(int).ravel()) 


print("-------------------")



# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].astype(int).ravel())
print("After OverSampling, counts of label '2': {}".format(sum(Y_train_keep == 2)))
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))
print(X_train_keep.shape)

numpy.random.shuffle(X_train_keep)

#=======================================
 
# Data Pre-processing - scale data using robust scaler

Y_train = X_train_keep[:, -1]
Y_test = X_test_tmp[:, -1]

print(Y_train)
print(Y_test)

input = X_train_keep[:, 0:32000]
testinput = X_test_tmp[:, 0:32000]

#=====================================

# Model configuration

input = input.reshape(len(input), 64, 500)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 64, 500)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)


# pick the list of hyperparameters from the csv file. 

hyperparameters = loadtxt('hyperparameters.csv', delimiter=',', skiprows=1)

hyperparameters_output = numpy.empty((len(hyperparameters), 9))

for i in range (0, len(hyperparameters)):

	hyperparameters_output[i, 0] = hyperparameters[i, 0]
	hyperparameters_output[i, 1] = hyperparameters[i, 1]
	hyperparameters_output[i, 2] = hyperparameters[i, 2]
	hyperparameters_output[i, 3] = hyperparameters[i, 3]
	hyperparameters_output[i, 4] = hyperparameters[i, 4]
	hyperparameters_output[i, 5] = 0
	hyperparameters_output[i, 6] = 0
	hyperparameters_output[i, 7] = 0
	hyperparameters_output[i, 8] = 0


	if(hyperparameters[i, 0] == 0):

		tensorflow.keras.backend.clear_session()

		seed(1)
		set_seed(1)

		# Create the model  ################## BEST ONE - SO FAR - 7 participants - 2 sec samples #######################

		my_L2 = hyperparameters[i, 1]
		positive_wt = hyperparameters[i, 2]
		negative_wt = hyperparameters[i, 3]

		model=Sequential()
		model.add(Conv1D(filters=65, kernel_size=6, kernel_regularizer=L2(my_L2), bias_regularizer=L2(my_L2), activity_regularizer = L2(my_L2), kernel_constraint=min_max_norm(min_value=negative_wt, max_value=positive_wt), padding='valid', activation='relu', strides=1, input_shape=(500, 64)))
		model.add(AveragePooling1D(pool_size=3)) 
		model.add(Conv1D(filters=65, kernel_size=6, kernel_regularizer=L2(my_L2), bias_regularizer=L2(my_L2), activity_regularizer = L2(my_L2), kernel_constraint=min_max_norm(min_value=negative_wt, max_value=positive_wt), padding='valid', activation='relu', strides=1))
		model.add(AveragePooling1D(pool_size=3)) 
		model.add(Conv1D(filters=24, kernel_size=5, kernel_regularizer=L2(my_L2), bias_regularizer=L2(my_L2), activity_regularizer = L2(my_L2), kernel_constraint=min_max_norm(min_value=negative_wt, max_value=positive_wt), padding='valid', activation='relu', strides=1))
		model.add(AveragePooling1D(pool_size=3)) 
		model.add(Conv1D(filters=24, kernel_size=5, kernel_regularizer=L2(my_L2), bias_regularizer=L2(my_L2), activity_regularizer = L2(my_L2), kernel_constraint=min_max_norm(min_value=negative_wt, max_value=positive_wt), padding='valid', activation='relu', strides=1))
		model.add(GlobalMaxPooling1D())
		model.add(Dense(3, activation='softmax'))

		model.summary()

		# Compile the model   
		adam = Adam(learning_rate=hyperparameters[i, 4])
		model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

		# simple early stopping
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
		mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

		hist = model.fit(input, Y_train, batch_size=32, epochs=60, verbose=0, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])

		#==================================

		model.save("model_conv1d.h5")

		# evaluate the model at the end of iterations
		_, train_acc_iter_end = model.evaluate(input, Y_train, verbose=1)
		_, test_acc_iter_end = model.evaluate(testinput, Y_test, verbose=1)
		print('Train: %.3f, Test: %.3f' % (train_acc_iter_end, test_acc_iter_end))

		# evaluate the model
		predict_y = model.predict(testinput)
		Y_hat_classes=numpy.argmax(predict_y,axis=-1)

		matrix = confusion_matrix(Y_test, Y_hat_classes)
		print(matrix)

		# load the best model
		saved_model = load_model('best_model.h5')
		# evaluate the best model
		_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
		_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
		print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

		# evaluate the model
		predict_y = saved_model.predict(testinput)
		Y_hat_classes=numpy.argmax(predict_y,axis=-1)

		matrix = confusion_matrix(Y_test, Y_hat_classes)
		print(matrix)

		#==================================	

		hyperparameters_output[i, 0] = hyperparameters[i, 0]
		hyperparameters_output[i, 1] = hyperparameters[i, 1]
		hyperparameters_output[i, 2] = hyperparameters[i, 2]
		hyperparameters_output[i, 3] = hyperparameters[i, 3]
		hyperparameters_output[i, 4] = hyperparameters[i, 4]
		hyperparameters_output[i, 5] = train_acc_iter_end
		hyperparameters_output[i, 6] = test_acc_iter_end
		hyperparameters_output[i, 7] = train_acc
		hyperparameters_output[i, 8] = test_acc

numpy.savetxt("hyperparameters_output_" + str(len(hyperparameters)) + ".csv", hyperparameters_output, delimiter=',')	