import tensorflow as tf
import tensorflow_probability as tfp
import math

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, lrf, nbatch, nw, warmup_bias_lr,epochs,cos_lr):
        super().__init__()
        self.lr0 = initial_learning_rate
        self.lrf=lrf # lr final
        self.nbatch=nbatch
        self.nb=nbatch # nof batches in epoch
        self.nw=nw # nof warmup steps
        self.lr = initial_learning_rate # redundant - lr value anyway initiated at __call__
        warmup_epochs = math.ceil(self.nw /self.nb) # (warmup steps)/(steps in epoch)
        if cos_lr:
            self.lf = lambda x:  0.5 * (1 + tf.cos((x - warmup_epochs) * math.pi / (epochs- warmup_epochs)))
        else: # linear decrement
            self.lf = lambda x: (1 - (x-warmup_epochs) /  (epochs- warmup_epochs) )* (1.0 - self.lrf) + self.lrf


    def __call__(self, step):
        if step == 0:
            self.epoch=0
        elif (step % self.nb) == 0:  # do per epoch
            self.epoch+=1
        if step <= self.nw: # warmup linear rise from 0.0 to lr0
            self.lr=tfp.math.interp_regular_1d_grid(
                    x=tf.cast(step, tf.float32),
                    x_ref_min = 0.,
                    x_ref_max=self.nw,
                    y_ref=[0, self.lr0]
                )
        elif (step % self.nb) == 0: # update each epoch
            self.lr =   self.lr0 * self.lf(self.epoch)
        return self.lr


