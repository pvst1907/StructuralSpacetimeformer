import copy
import numpy as np
import torch
import torch.nn as nn
from utils import Accumulator, Animator


def train_epoch(xformer,
                train_iter,
                loss,
                encoder_optimizer,
                decoder_optimizer):
    xformer.train()
    tracker = Accumulator(2)
    for src, trg, trg_y in train_iter:
        trg_y_hat = xformer(src, trg)
        l = loss(trg_y.transpose(2,1), trg_y_hat)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        l.mean().backward()
        #nn.utils.clip_grad_norm_(xformer.parameters(), 3)
        nn.utils.clip_grad_value_(xformer.parameters(), 3)
        encoder_optimizer.step()
        decoder_optimizer.step()

    tracker.add(float(l.sum()), trg_y.numel())
    return tracker[0] / tracker[1]


def train_torch(xformer,
                train_iter,
                test_iter,
                loss,
                metric,
                epochs,
                encoder_optimizer,
                decoder_optimizer,
                patience=100,
                verbose=False,
                plot=False):
    min_loss = np.nan
    epochs_no_improve = 0
    #torch.autograd.set_detect_anomaly(True)
    if plot:
        animator = Animator(xlabel='epoch', xlim=[1, epochs], legend=['training loss', 'evaluation_loss'])
    for epoch in range(epochs):
        train_tracker = train_epoch(xformer, train_iter, loss, encoder_optimizer, decoder_optimizer)
        eval_tracker = evaluate_model(xformer, test_iter, metric)
        epochs_no_improve += 1
        if np.isnan(min_loss) or train_tracker < min_loss:
            min_loss = train_tracker
            state_dict = copy.deepcopy(xformer.state_dict())
            epochs_no_improve = 0
        if epoch > 5 and epochs_no_improve > patience:
            xformer.load_state_dict(state_dict)
            print('Stopped at ', epoch, train_tracker)
            break
        if plot:
            animator.add(epoch + 1, (train_tracker, eval_tracker))
        if verbose:
            print(train_tracker)
    return train_tracker, eval_tracker


def evaluate_model(xformer, data_iter, metric):
    xformer.eval()
    tracker = Accumulator(2)
    with torch.no_grad():
        for src, trg, trg_y in data_iter:
            tracker.add(float(metric(xformer(src, trg), trg_y).sum()), trg_y.numel())
    return tracker[0] / tracker[1]



