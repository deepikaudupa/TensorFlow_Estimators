#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""



#--------
# source trialevn/bin/activate
# pip install matplotlib
# pip install opencv-contrib-python
# pip install tensorflow==1.14.*
# gcloud ai-platform local train --module-name trainer.task --package-path trainer/ -- --eval-steps 5
# gcloud ai-platform local train --module-name trainer.final_task --package-path trainer/ -- --eval-steps 5
#--------

#model name: model_findml6mug
#MODEL_NAME=model_findml6mug
#gcloud ai-platform models create $MODEL_NAME --regions=$REGION

##MODEL_BINARIES=gs://$BUCKET_NAME/output/export/exported/1579188679/


 # gcloud ai-platform versions create v1 \
 #     --model $MODEL_NAME \
 #     --origin $MODEL_BINARIES \
 #     --runtime-version 1.14


#gcloud ai-platform predict \ --model $MODEL_NAME \ --version v1 \  --json-instances /test.json
#gcloud ai-platform predict --model $MODEL_NAME --version v1 --json-instances check_deployed_model/test.json



import argparse
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.data as data
import trainer.model as model


def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the test folder and trains your solution from the model.py file with it."""
    (train_data, train_labels) = data.create_data_with_labels("data/train/")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=model.get_batch_size(),
        num_epochs=None,
        shuffle=True)

    (eval_data, eval_labels) = data.create_data_with_labels("data/test/")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    estimator = tf.estimator.Estimator(model_fn=model.solution)

    steps_per_eval = int(model.get_training_steps() / params.eval_steps)

    for _ in range(params.eval_steps):
        estimator.train(train_input_fn, steps=steps_per_eval)
        estimator.evaluate(eval_input_fn)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=1,
        type=int
    )

    ARGS = PARSER.parse_args()
    tf.logging.set_verbosity('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)

    HPARAMS = hparam.HParams(**ARGS.__dict__)
    train_model(HPARAMS)
