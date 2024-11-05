import math
import sys
import torch
from typing import Iterable

import utils.misc as misc
import utils.lr_sched as lr_sched
from engines.utils import update_ema, log_train_state, log_visuals

def train_one_epoch(model: torch.nn.Module, ema: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler,
        log_writer=None, args=None):
    
    ema.eval()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    optimizer.zero_grad()
    update_ema(ema, model.module, decay=0.)

    for data_iter_step, (input_samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        input_samples = input_samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(): # dtype=torch.bfloat16
            loss, train_logs = model(input_samples)
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            update_ema(ema, model.module)

        torch.cuda.synchronize()

        # Sysout logging nll_loss for vqgan base tokenizer, and code loss for vae base tokenizer
        for iter_logs_dict in train_logs:
            for key in iter_logs_dict.keys():
                # if "nll_loss" in key:
                if "nll_loss" in key or "code_loss" in key:
                    metric_logger.update(**{key: iter_logs_dict[key]})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(torch.tensor(loss_value).cuda()).item()
    
        if misc.get_rank()==0 and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            base_logs = {'train_loss': loss_value_reduce, 'lr':lr}

            log_train_state(
                base_logs, train_logs, epoch_1000x, 
                log_writer, tensorboard_logging=False, wandb_logging=True)

        if misc.get_rank()==0 and data_iter_step==0:
            # Visualizing first 16 reconstructed images of the trainset only.
            # To visualize on a validation set, simply replace data_loader with validation data_loader.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            
            with torch.no_grad():
                visualization_logs = ema(input_samples, reconstruction_iters="all")
                log_visuals(
                    input_samples, visualization_logs, epoch_1000x,
                    log_writer, tensorboard_logging=False, wandb_logging=True)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}