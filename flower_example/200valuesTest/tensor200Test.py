from __future__ import absolute_import, division, print_function

import sys
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe



#function for prepareing the train,test and prediction-data sets
def parse_csv(line):
  example_defaults = [
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
  [0]]  
  # this sets the field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 200 fields are features, combine into single tensor
  # transpose first 200 elements to column vector
  features = tf.reshape(parsed_line[:-1], shape=(200,))
  # Last field is the label
  # goes into seperate 1x1 vector -> scalar
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

#functions for training the network
#loss determinates how bad the model is performing
    # tf.losses.sparse_softmax_cross_entropy function takes the model's prediction and the desired label. 
    #The returned loss value is progressively larger as the prediction gets worse.
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

#grad uses the loss function and the tf.GradientTape to record operations that compute the gradients used to optimize our model.
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)





#initialize
tf.enable_eager_execution()
#Print that import is done and tf is ready
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#Error handling for missing Dataset-Files
missingParMsg ='\nEnter Filename for Training, Test, and Prediction-Dataset-CSV!\n To run the script command:\n python tensor200Test.py trainingDataFile.csv testDataFile.csv predictionDataFile.csv' 
try:
    sys.argv[1]
except IndexError:
    raise FileNotFoundError(missingParMsg)
try:
    sys.argv[2]
except IndexError:
    raise FileNotFoundError(missingParMsg)
#try:
#    sys.argv[3]
#except IndexError:
#    raise FileNotFoundError(missingParMsg)


#IMPORT THE DATA
#import the training Data
trainingDataFilename = sys.argv[1]
train_dataset_fp = os.getcwd()+'/'+trainingDataFilename
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)
print('competed reading Training-Data')

#read the test Data
testDataFilename = sys.argv[2]
test_fp = os.getcwd()+'/'+ testDataFilename
test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set
print('competed reading Test-Data')

#read the predict Data
#predDataFilename = sys.argv[3]
#pred_fp  = os.getcwd()+'/'+ predDataFilename
#predict_dataset  = tf.data.TextLineDataset(pred_fp)
#predict_dataset  = predict_dataset.map(parse_csv)      # parse each row with the funcition created earlier
#predict_dataset = predict_dataset.batch(32)           # use the same batch size as the training set
#print('competed reading Prediction-Data')


#DEFINING THE MODEL OF THE NETWORK
model = tf.keras.Sequential([
    #creating 2 hidden layers
        #An activation function with the following rules:
            #If input is negative or zero, output is 0.
            #If input is positive, output is equal to input.
    tf.keras.layers.Dense(400, activation="relu", input_shape=(200,)),  # input shape required
    tf.keras.layers.Dense(400, activation="relu"),
    #creating a output layers, use softmax in last layer to represent probabilities 
    tf.keras.layers.Dense(3, activation="softmax")
])
print('crated model network.')


#TRAIN THE NETWORK
#set up the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# keep results for plotting - TODO: shall we use this?
train_loss_results = []
train_accuracy_results = []

#defining the numer of leraning-iteraions (calles epoches here)
num_epochs = 201

#do the actual training
for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  #print progress
  if epoch % 10 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


#EVALUATE THE MODEL
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



#PREDICTION
#class_ids = ["raph", "dennis"]
#
#for x, y in predict_dataset:
#  #just use the 200 values, reject the seted label 
#  predictions = model(x)
#  #print results for every Run-part
#  for idx,val in enumerate(predictions.numpy()):
#    print("Prediction for Run-part {} ==> {}::   Raph:{:.3%}%, Dennis:{:.3%}%".format(idx,class_ids[tf.argmax(val).numpy() ] ,val[0],val[1]))
#