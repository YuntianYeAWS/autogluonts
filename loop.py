# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, Optional, Union

# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.gluonts_tqdm import tqdm

from gluonts.support.util import HybridContext

# Relative imports
from gluonts.trainer import learning_rate_scheduler as lrs
from asset import *

epochs = 10

trainer = mx.gluon.Trainer(
                    net.collect_params(),
                    optimizer=optimizer,
                    kvstore="device",  # FIXME: initialize properly
                )
avg_strategy = AveragingStrategy()
def loop(
        epoch_no, batch_iter, is_training: bool = True
):
    print('check 0')
    tic = time.time()

    epoch_loss = mx.metric.Loss()

    # use averaged model for validation
    if not is_training and isinstance(
            avg_strategy, IterationAveragingStrategy
    ):
        avg_strategy.load_averaged_model(net)

    print('check 1')
    with tqdm(batch_iter) as it:
        for batch_no, data_entry in enumerate(it, start=1):

            if False:
                break

        inputs = [data_entry[k] for k in input_names]
        print('SOMETHING HERE')

        with mx.autograd.record():
            output = net(*inputs)

            # network can returns several outputs, the first being always the loss
            # when having multiple outputs, the forward returns a list in the case of hybrid and a
            # tuple otherwise
            # we may wrap network outputs in the future to avoid this type check
            if isinstance(output, (list, tuple)):
                loss = output[0]
            else:
                loss = output
        print('check 2')
        if is_training:
            loss.backward()
            trainer.step(batch_size)

            # iteration averaging in training
            if isinstance(
                    avg_strategy,
                    IterationAveragingStrategy,
            ):
                avg_strategy.apply(net)

        epoch_loss.update(None, preds=loss)
        lv = loss_value(epoch_loss)
        print('check 3')
        if not np.isfinite(lv):
            logger.warning(
                "Epoch[%d] gave nan loss", epoch_no
            )
            return epoch_loss

        it.set_postfix(
            ordered_dict={
                "epoch": f"{epoch_no + 1}/{epochs}",
                ("" if is_training else "validation_")
                + "avg_epoch_loss": lv,
            },
            refresh=False,
        )
        # print out parameters of the network at the first pass
        if batch_no == 1 and epoch_no == 0:
            net_name = type(net).__name__
            num_model_param = count_model_params(net)
            logger.info(
                f"Number of parameters in {net_name}: {num_model_param}"
            )
    return inputs