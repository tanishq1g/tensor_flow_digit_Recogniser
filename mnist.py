# future for running the code on python 2 and import the code from python2 to python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    ## input layer
    # input tensors should have a shape of [batch_size, image_height, image_width, channels] by default for the other layers to use them
    # tf.reshape(tensor, shape, name=None)
    # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    input_layer = tf.reshape(features['x'], [-1,28,28,1], name="intial name")


    ## Convolutional Layer #1
    # 32 5*5 filters applied with ReLU activation fu
    conv1 = tf.layers.conv2d(  # output tensor - [batch_size, 28, 28, 32]
        inputs = input_layer,  #tensor input
        filters = 32,          #no of filters
        kernel_size = [5, 5],  #h and w of the 2D conv window
        padding = "same",      #zero padding added to preserve h and w of 28- output tensor should have the same h and w as the input tensor
        activation = tf.nn.relu, #activation func - max(features,0)
        name="convulation layer 1"
    )

    ## Pooling Layer #1
    # input tensor should have the shape [batch_size, image_height, image_width, channels] which here is [batch_size, 28, 28, 32]
    # pool size - size of max pooling filter
    # strides - no of pixels between each recepetive field
    # OUTPUT tensor - [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2, name = "pooling layer 1")

    ## Convolutional Layer #2
    conv2 = tf.layers.conv2d( #output tensor - [batch_size, 14, 14, 64]
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu,
        name="convulation layer 2"
    )

    ## Pooling Layer #2
    # output tensor - [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2, name = 'pooling layer 2')

    ## Dense Layer
    #flat out input to [batch_size, features] i.e. [batch_size, 3136]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name="flatting layer pool")
    #units - number of neurons in dense layer,1024 ??
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu, name = "dense layer 1")
    # dense - [batch_size, 1024]
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN, name = "dropout ")
    # dropout - [batch_size, 1024]

    ## Logits Layer
    # return the raw values for our predictions. We create a dense layer with 10 neurons (one for each target class 0–9), with linear activation (the default)
    # logits - [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input = logits, axis = 1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels = labels, predictions = predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(  #wraps the model given by model_fn( gives inputs and other paramters ) and the function returns necesaary operations necesaary for training, predictions and evaluations
        model_fn = cnn_model_fn,
        model_dir = "/yup/machine learning/oriley_hands_on_ml/MNIST/mnist_logs"  #all outputs are written to this dir
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log,   #dict that maps the tenosr to be printed
        every_n_iter = 50           #prints the values every 50 iterations
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(  #Returns input function that would feed dict of numpy arrays into the model.
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        hooks = [logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    # print (eval_results)


if __name__ == "__main__":
    tf.app.run()
