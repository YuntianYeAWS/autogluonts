import autogluon as ag
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from dataset import dataset
from gluonts.trainer import Trainer
args = ag.space.Dict(lr = ag.space.Real(lower = 1e-3,upper = 1e-2),
                  epochs=ag.space.Choice(40, 80),
                  hid_dim = ag.space.Int(lower = 3, upper = 10) )

def train_finetune(args, reporter):
    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[args.hid_dim],
        prediction_length=dataset.metadata.prediction_length,
        context_length=100,
        freq=dataset.metadata.freq,
        trainer=Trainer(ctx="cpu",
                        epochs=5,
                        learning_rate=args.lr,
                        num_batches_per_epoch=100
                        )
    )
    for epoch in range(args.epochs):
        reporter(epoch=epoch + 1, accuracy=1)

myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                         resource={'num_cpus': 4, 'num_gpus': 0},
                                         num_trials=5,
                                         time_attr='epoch',
                                         reward_attr="accuracy")
print(myscheduler)