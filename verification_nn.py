from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
from numpy.random import seed
from tensorflow.random import set_seed


# setting the seed
seed(1)
set_seed(1)

# load the test data
X = loadtxt('loaded_complete_unseen_data_2_sec.csv', delimiter=',')

input = X[:, 0:32000]

# transform the data in the format which the model wants
input = input.reshape(len(input), 64, 500)
input = input.transpose(0, 2, 1)

# get the expected outcome 
y_real = X[:, -1]

# load the model
model = load_model('best_model.h5')

# get the "predicted class" outcome
predict_y = model.predict(input)
Y_hat_classes=numpy.argmax(predict_y, axis=-1)

matrix = confusion_matrix(y_real, Y_hat_classes)
print(matrix)
