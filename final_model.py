#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf


#Define the parameters
TRAINING_STEPS= 20000  #TRIED VALUES: 20000/25000/30000. 20000: TRADE OFF BETWEEN ACCURACY AND TRAINING TIME.
BATCH_SIZE= 20 
LEARNING_RATE= 0.0001 
DROPOUT_RATE_LOGITS= 0.4  
#DROPOUT_RATE_LAYERS= 0.25 #DROPOUT FOR INTERMEDIATE LAYERS. 75% CHANCE OF SELECTION 
#NUMBER OF CLASSES= 4

def get_training_steps():
    return TRAINING_STEPS


def get_batch_size():
    return BATCH_SIZE


#EACH CLASS HAS ONLY 250 IMAGES AS TRAINING SET. THAT IS NOT ENOUGH DATA TO TRAIN A CNN. 
#TO ACHIEVE BETTER ACCURACY, ONE CAN IMPLEMENT AUGMENTATION TO ACHIEVE MORE TRAINING SAMPLE. 
# def augmented_input(input_layer):
#     num = input_layer.get_shape().as_list()[0]

#     # random_brightness= tf.image.random_brightness(
#     #                     input_layer,
#     #                     max_delta=0.6
#     #                     )
#     invert_horizontal = tf.image.random_flip_left_right(input_layer)
#     invert_vertical = tf.image.random_flip_up_down(invert_horizontal)
#     #MORE AUGMENTATIONS TRIED. RANDOMNOISE, CROP ETC. NO IMPROVEMENT IN ACCURACY.
#     return invert_vertical


def solution(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3]) 


    # CONVOLUTION LAYER 1 : 2D convolution layer ...
    conv2d1= tf.layers.conv2d(
        inputs=input_layer, #INPUT: 64X64X3 [BATCH_SIZE, 64, 64, 3] 3 BEING RGB
        filters=32, #NUMBER OF FEATURES: N=32
        kernel_size= [5,5], # KERNEL SIZE IS DEFINED AS : [nxn]=[5X5] FILTER.
        padding="same", # PADDING ADDED TO THE WIDTH AND HEIGHT. output image size is same as input.
        activation= tf.nn.relu # ACTIVATION USED : ReLU
        )

    # POOLING LAYER 1: 2D MAXPOOLING LAYER
    # MAXPOOLING: Downsampling with a [mxm] filter with M stride 
    # OUTPUT : 32X32X32 [BATCH_SIZE, 32, 32, 32]
    maxpool1= tf.layers.max_pooling2d(
        inputs=conv2d1, 
        pool_size=[2, 2], #Size of the pooling window
        strides=2 #factor of downscaling. Stride=2 will half the the input. 
        )


    # CONVOLUTION LAYER 2 : 2D convolution layer
    # OUTPUT: 32X32X64 [BATCH_SIZE, 32, 32, 64]
    conv2d2= tf.layers.conv2d(
        inputs= maxpool1, #INPUT: 32X32X32 [BATCH_SIZE, 32, 32, 32]
        filters=64, #NUMBER OF FEATURES: 64 (SAME AS THE SIZE OF THE IMAGE)
        kernel_size= [5,5], # KERNEL SIZE IS DEFINED AS : 5X5 FILTER.
        padding="same", # PADDING ADDED TO THE WIDTH AND HEIGHT. output image size is same as input.
        activation= tf.nn.relu # ACTIVATION USED : ReLU
        )

    # POOLING LAYER 2: 2D MAXPOOLING LAYER
    # OUTPUT : 16X16X64 [BATCH_SIZE, 16, 16, 64]
    maxpool2= tf.layers.max_pooling2d(
        inputs=conv2d2, #input: 32X32X64 [BATCH_SIZE, 32, 32, 64]
        pool_size=[2, 2], #Size of the pooling window
        strides=2 #factor of downscaling. Stride=2 will half the the input. 
        )

    #FLATTENING TO A 1-D VECTOR FOR THE FULLY CONNECTED DENSE LAYER
    flat_layer= tf.contrib.layers.flatten(maxpool2)


    # FULLY CONNECTED DENSE LAYER
    fc_layer1 = tf.layers.dense(
        inputs=flat_layer, 
        units=1024, #NUMBER OF NEURONS= 1024
        activation=tf.nn.relu # ACTIVATION USED : ReLU
        )

    # ADD DROPOUT TO PREVENT OVER FITTING. 
    dropout_fin = tf.layers.dropout(
        inputs=fc_layer1, 
        rate=DROPOUT_RATE_LOGITS, # 60% PROBABILITY THAT THE VALUE WILL BE SELECTED. 
        training= mode == tf.estimator.ModeKeys.TRAIN #ACTIVATES DROPOUT ONLY WHEN MODE=TRAIN
        )

    # FULLY CONNECTED DENSE LAYER : OUTPUT LAYER: Logits!
    output_layer = tf.layers.dense(
        inputs=dropout_fin, 
        units=4, #NUMBER OF NEURONS= 4 : for the 4 classes
        )

    
    # define the predictions first
    predictions = { 
        "classes": tf.argmax(input=output_layer, axis=1), 
        "probabilities" : tf.nn.softmax(output_layer, name="softmax_tensor") 
        # THIS IS TO ACHIEVE THE MULTICLASS CLASSIFICATION. USED IN PREDICT STEP. 
        #THE CLASS WITH THE HIGHEST PROBABILITY WILL BE CHOSEN AS LABEL. 
        }

    

    # DEFINING ACTIONS FOR THE THREE MODES OF ESTIMATOR
    if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
            #DEFINE LOSS FUNCTION FOR MODE: EVAL and TRAIN
            labels= tf.cast(labels, tf.int64)
            #tf.print(labels)
    
            loss_fn= tf.losses.sparse_softmax_cross_entropy(
                labels=labels, 
                logits=output_layer
                )
            loss_fn= tf.reduce_mean(loss_fn)

            # DEFINE OPTIMIZER FOR TRAIN! 
            # TRIED. GRADIENT DECENT OPTIMIZER/ STOCHASTIC GD/ ADAM. HIGHEST ACCURACY WAS FOUND WITH ADAM. 
            optimizer_fn= tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)            
            #optimizer_fn= tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            
            train_op= optimizer_fn.minimize(
                loss=loss_fn,
                global_step=tf.train.get_global_step()
                )

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_fn, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL: 

            #DEFINE LOSS FUNCTION FOR MODE: EVAL and TRAIN
            labels= tf.cast(labels, tf.int64)
            #tf.print(labels)
    
            loss_fn= tf.losses.sparse_softmax_cross_entropy(
                labels=labels, 
                logits=output_layer
                )
            loss_fn= tf.reduce_mean(loss_fn) 
            metrics = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
                }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_fn, eval_metric_ops=metrics)
