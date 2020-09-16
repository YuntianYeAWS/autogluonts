import os
import numpy as np

import mxnet as mx
from gluonts.trainer import learning_rate_scheduler as lrs
from mxnet import gluon, init
from newloop import newloop
import autogluon as ag
from autogluon.utils.mxutils import get_data_rec
from loop import loop
from testloop import training_data_loader
from estimator import estimator
from gluonts.gluonts_tqdm import tqdm
input_names = ['past_target', 'future_target']
from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator
from dataset import dataset
from gluonts.trainer import Trainer
from asset import optimizer
with tqdm(training_data_loader) as it:
    for batch_no, data_entry in enumerate(it, start=1):

        if False:
            break

    inputs = [data_entry[k] for k in input_names]

dictionary_of_hyperparameters = {}
dictionary_of_hyperparameters ['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)
dictionary_of_hyperparameters['epochs']=ag.Choice(40, 80)

@ag.args()
def train_finetune(args, reporter):
    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=dataset.metadata.prediction_length,
        context_length=100,
        freq=dataset.metadata.freq,
        trainer=Trainer(ctx="cpu",
                        epochs=5,
                        learning_rate=args.learning_rate,
                        num_batches_per_epoch=100
                        )
    )
    net = estimator.create_training_network()
    net.initialize(ctx=None, init='xavier')
    lr_scheduler = lrs.MetricAttentiveScheduler(
        objective="min",
        patience=estimator.trainer.patience,
        decay_factor=estimator.trainer.learning_rate_decay_factor,
        min_lr=estimator.trainer.minimum_learning_rate,
    )
    optimizer = mx.optimizer.Adam(
        learning_rate=estimator.trainer.learning_rate,
        lr_scheduler=lr_scheduler,
        wd=estimator.trainer.weight_decay,
        clip_gradient=estimator.trainer.clip_gradient,
    )
    trainer = mx.gluon.Trainer(
        net.collect_params(),
        optimizer=optimizer,
        kvstore="device",  # FIXME: initialize properly
    )
    print('Learning rate: ', args.lr)
    for epoch in range(args.epochs):
        mse = newloop(epoch,net,trainer,inputs)
        print('MSE:', mse)
        reporter(epoch = epoch+1, accuracy = -mse)

train_finetune.register_args(**dictionary_of_hyperparameters)
myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                         resource={'num_cpus': 4, 'num_gpus': 0},
                                         num_trials=5,
                                         time_attr='epoch',
                                         reward_attr="accuracy")

print(myscheduler)
