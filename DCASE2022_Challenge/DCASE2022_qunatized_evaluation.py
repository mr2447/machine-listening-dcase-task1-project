import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten,
    BatchNormalization,
    Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
)

from sklearn.metrics import log_loss
#%%
BASE_PATH = os.path.expanduser('~/Documents/MListening_project/DCASE2022_Challenge/DCASE2022_numpy_mel_train_test_data')

x_train = np.reshape(
    np.load(os.path.join(BASE_PATH, 'DCASE2022_train.npy')),
    [139620, 40, 51, 1]
)

x_test = np.reshape(
    np.load(os.path.join(BASE_PATH, 'DCASE2022_test.npy')),
    [29680, 40, 51, 1]
)

labels_train = np.load(os.path.join(BASE_PATH, 'label_train.npy'))
labels_test = np.load(os.path.join(BASE_PATH, 'label_test.npy'))
y_test = tf.keras.utils.to_categorical(labels_test, 10)
y_train=  tf.keras.utils.to_categorical(labels_train, 10)

#%%

os.chdir('Quantized_model/C123')   # set path of the quantized model.


prob_def = [ ]
interpreter = tf.lite.Interpreter(model_path="converted_quant_model_default.tflite")


all_tensor_details=interpreter.get_tensor_details()
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
# print(input_shape)



x_test=np.array(x_test,dtype=np.float32)
acc=0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob_def.append(output_data)
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc_def = acc/len(x_test)
# print('default compressed acc:',acc_def*100)
logloss_def= log_loss(y_true=labels_test, y_pred= np.reshape(prob_def,[-1,10]), normalize=True)
np.save(os.path.join(os.getcwd(), 'prob_def.npy'), np.reshape(prob_def, [-1, 10]))



print('..............................')
print('INT8: : ',acc_def*100,'logloss:  ', logloss_def)

print('..............................')

prob = np.load('prob_def.npy')
print("Loaded prediction shape:", prob.shape)
print("First 5 predictions:\n", prob[:5])

predicted_classes = np.argmax(prob, axis=1)
print("First 5 predicted classes:", predicted_classes[:5])