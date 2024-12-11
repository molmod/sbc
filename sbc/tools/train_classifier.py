###########################################################################################
# Training script for classifier block
# Author: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
#from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader

import mace.tools.torch_geometric as torch_geometric
#from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.torch_tools import to_numpy


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    eval_interval: int,
    max_num_epochs: int,
    patience: int,
    device: torch.device,
    max_grad_norm: Optional[float] = 10.0,
):
    lowest_loss = np.inf
    valid_loss  = np.inf
    lowest_loss_epoch = 0

    highest_acc = 0
    valid_acc   = 0
    highest_acc_epoch = 0

    patience_counter = 0
    swa_start = True
    keep_last = False

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
    logging.info("Started training")
    epoch = 0
    while epoch < max_num_epochs:
        lr_scheduler.step(
            metrics=valid_loss
        )  # Can break if exponential LR, TODO fix that!

        # Train
        for batch in train_loader:
            _, opt_metrics = take_step(
                model=model,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch

        # Validate
        if epoch % eval_interval == 0:
            valid_loss, valid_acc, eval_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                data_loader=valid_loader,
                device=device,
            )
            s = "Epoch {:4}: loss={:9.4f}  ".format(epoch, valid_loss)
            for phase in model.classifier.phases:
                key = 'acc({})'.format(phase)
                value = eval_metrics[key]
                if value is not None:
                    s += '{}: {:4.3f}'.format(key, value)
                s += ' | '
            for key in ['CE', 'RMSE(e)']: 
                s += '{}: {:5.3f}'.format(key, eval_metrics[key])
                s += ' | '
            #s += ' | gd: {:5.1f}'.format(eval_metrics['gd'] * 1e3)
            #s += ' | gm: {:5.1f}'.format(eval_metrics['gm'] * 1e3)
            logging.info(s)

            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(
                        f"Stopping optimization after {patience_counter} epochs without improvement"
                    )
                    logging.info('lowest loss: {}'.format(lowest_loss))
                    break
            else:
                lowest_loss_epoch = epoch
                torch.save(model, 'lowest_loss.pth'.format(epoch))
                lowest_loss = valid_loss
                patience_counter = 0
            if valid_acc > highest_acc:
                highest_acc_epoch = epoch
                torch.save(model, 'highest_acc.pth'.format(epoch))
                highest_acc = valid_acc
        epoch += 1
    logging.info("Training complete")
    logging.info("Saved best loss model from epoch {} with loss {}".format(
        lowest_loss_epoch, lowest_loss))
    logging.info("Saved highest accuracy model from epoch {} with accuracy {}".format(
        highest_acc_epoch, highest_acc))


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()
    cls_data = dict(
        batch=batch_dict['batch'],
        ptr=batch_dict['ptr'],
        node_feats=batch_dict['node_feats'],
    )
    output = model.classifier(cls_data)
    loss = loss_fn(output=output, ref=batch)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict[str, Any]]:
    logits_computed = False
    accuracies = {p: None for p in model.classifier.phases}
    logits_list = []
    phase_list = []
    energies_list = []

    start_time = time.time()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        cls_data = dict(
            batch=batch_dict['batch'],
            ptr=batch_dict['ptr'],
            node_feats=batch_dict['node_feats'],
        )
        output = model.classifier(cls_data)

        loss = loss_fn(output=output, ref=batch)
        total_loss += to_numpy(loss).item()

        logits_computed = True
        logits_list.append(output['logits'])
        phase_list.append(batch['phase'])
        energies_list.append(output['node_delta'])

    avg_loss = total_loss / len(data_loader)

    aux = {
        "loss": avg_loss,
    }

    logits = torch.cat(logits_list, dim=0)
    phases = torch.cat(phase_list, dim=0).to(torch.long)
    energies = torch.cat(energies_list, dim=0)
    prediction = torch.argmax(logits, dim=1)
    true_p_sum = 0
    total_sum = 0
    for i, phase in enumerate(model.classifier.phases):
        mask = torch.eq(phases, torch.tensor([i]).to('cuda'))
        total = torch.sum(mask)
        if total > 0:
            true_p = torch.sum(torch.eq(prediction[mask], torch.tensor([i]).to('cuda')))
            total_sum += total
            true_p_sum += true_p
            accuracies[phase] = true_p / total
        aux['acc({})'.format(phase)] = accuracies[phase]
    valid_acc = true_p_sum / total_sum
    aux['CE'] = loss_fn.ce_loss(logits, phases)
    aux['RMSE(e)'] = torch.sqrt(torch.mean(torch.square(energies)))

    aux["time"] = time.time() - start_time
    return avg_loss, valid_acc, aux
