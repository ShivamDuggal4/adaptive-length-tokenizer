import math
import sys
import torch
from typing import Iterable

import utils.misc as misc
import utils.lr_sched as lr_sched
from engines.utils import update_ema, log_train_state, log_visuals

def train_one_epoch(model: torch.nn.Module, ema: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, loss_scaler_discriminator,
        log_writer=None, args=None):
    
    ema.eval()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_discriminator.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    update_ema(ema, model.module, decay=0.)

    max_iters = 8
    current_repeatition_iter = 0
    for data_iter_step, (input_samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch<=20: gan_loss_weight=0.0
        elif epoch<=100: gan_loss_weight=0.2
        else: gan_loss_weight=0.8
        
        gan_optimizer_idx = 1-((data_iter_step+1)%(max_iters+1)==0)*1
        if gan_loss_weight==0: gan_optimizer_idx=0
        
        if data_iter_step % accum_iter == 0:
            if gan_optimizer_idx==0: lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            if gan_optimizer_idx==1: lr_sched.adjust_learning_rate(optimizer_discriminator, data_iter_step / len(data_loader) + epoch, args)

        input_samples = input_samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, train_logs = model(
                input_samples, sample_grad_iters=[current_repeatition_iter%max_iters],
                gan_optimizer_idx=gan_optimizer_idx, gan_loss_weight=gan_loss_weight
            )
        
        if gan_optimizer_idx==0: current_repeatition_iter += 1
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if gan_optimizer_idx==1:
            loss_scaler_discriminator(loss, optimizer_discriminator, clip_grad=args.grad_clip, parameters=model.module.gan_losses.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            all_params = []
            for name, param in model.named_parameters():
                if "gan_losses" not in name:
                    all_params.append(param)
            loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=all_params,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()
            update_ema(ema, model.module)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        lr_disc = optimizer_discriminator.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(lr_disc=lr_disc)
        loss_value_reduce = misc.all_reduce_mean(torch.tensor(loss_value).cuda()).item()    
    
        if misc.get_rank()==0 and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            base_logs = {'train_loss': loss_value_reduce, 'lr': lr, 'lr_disc': lr_disc}

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