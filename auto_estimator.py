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

class AutoEstimator:
    def __init__(self,search_space):
        search_config = {}
        search_config['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)
        search_config['epochs'] = ag.Choice(40, 80)
        for config in search_config.keys():
            if not config in search_space.keys():
                search_space[config] = search_config[config]
        self.search_space = search_space



    def train(self):
        @ag.args(
            lr=ag.space.Real(1e-3, 1e-2, log=True),
            epochs=10)
        def train_finetune(args, reporter):
            estimator = SimpleFeedForwardEstimator(
                num_hidden_dimensions=[10],
                prediction_length=dataset.metadata.prediction_length,
                context_length=100,
                freq=dataset.metadata.freq,
                trainer=Trainer(ctx="cpu",
                                epochs=5,
                                learning_rate=args.lr,
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
            for epoch in range(args.epochs):
                mse = newloop(epoch, net, trainer, inputs)
                print('MSE:', mse)
                reporter(epoch=epoch + 1, accuracy=-mse)


        self.scheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                                 resource={'num_cpus': 4, 'num_gpus': 0},
                                                 num_trials=5,
                                                 time_attr='epoch',
                                                 reward_attr="accuracy")

    def create_data(self,dataset):
        input_names = ['past_target', 'future_target']
        training_data_loader = TrainDataLoader(
            dataset=dataset.train,
            transform=transformation,
            batch_size=trainer.batch_size,
            num_batches_per_epoch=trainer.num_batches_per_epoch,
            ctx=trainer.ctx,
            dtype=dtype,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
        )
        with tqdm(training_data_loader) as it:
            for batch_no, data_entry in enumerate(it, start=1):
                if False:
                    break
            inputs = [data_entry[k] for k in input_names]



