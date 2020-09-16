from loop import loop
from newloop import newloop
from dataset import dataset
from estimator import estimator
from gluonts.dataset.loader import TrainDataLoader
from gluonts.trainer import Trainer
import numpy as np
from estimator import net
from gluonts.gluonts_tqdm import tqdm

training_data = dataset.train
transformation = estimator.create_transformation()
dtype = np.float32
num_workers = None
num_prefetch = None
shuffle_buffer_length = None
trainer=Trainer(ctx="cpu",
                epochs=1,
                learning_rate=0.01,
                num_batches_per_epoch=100
               )
training_data_loader = TrainDataLoader(
        dataset=training_data,
        transform=transformation,
        batch_size=trainer.batch_size,
        num_batches_per_epoch=trainer.num_batches_per_epoch,
        ctx=trainer.ctx,
        dtype=dtype,
        num_workers=num_workers,
        num_prefetch=num_prefetch,
    )
input_names = ['past_target', 'future_target']
with tqdm(training_data_loader) as it:
    for batch_no, data_entry in enumerate(it, start=1):

        if False:
            break

    inputs = [data_entry[k] for k in input_names]

net.initialize(ctx=None, init='xavier')


if __name__ == '__main__':

   # loop(1,training_data_loader)
    newloop(1,inputs)


