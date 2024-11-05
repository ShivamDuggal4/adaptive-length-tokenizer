import torch
import wandb
from collections import OrderedDict
from einops import repeat, rearrange

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def log_train_state(base_logs, train_logs, epoch_1000x, log_writer=None, tensorboard_logging=True, wandb_logging=True):
    if tensorboard_logging:
        assert(log_writer is not None)
        for key, value in base_logs.items():
            log_writer.add_scalar(key, value, epoch_1000x)
    
    if wandb_logging:
        wandb_log_data = base_logs
        for iter_logs_dict in train_logs:
            for key in iter_logs_dict.keys():
                if "reconstructed_imgs" in key: continue
                wandb_log_data[key] = iter_logs_dict[key]
        wandb.log(wandb_log_data, step=epoch_1000x)

def log_visuals(input_samples, visualization_logs, epoch_1000x, log_writer=None, tensorboard_logging=True, wandb_logging=True, num_images_to_log=16):
    for iter_logs_dict in visualization_logs:
        for key in iter_logs_dict.keys():
            if "reconstructed_imgs" not in key: continue
            
            # Make sure num_images_to_log is factor of 4.
            input_imgs = input_samples[:num_images_to_log]
            reconstructed_imgs = iter_logs_dict[key][:num_images_to_log]
            vis_img = torch.cat([input_imgs.detach().cpu(), reconstructed_imgs.detach().cpu()], dim=0)
            vis_img = rearrange(vis_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=2)
            
            if tensorboard_logging:
                assert(log_writer is not None)
                log_writer.add_image('train-input-{}'.format(key), (vis_img + 1) / 2, global_step=epoch_1000x)
            
            if wandb_logging:
                wandb_images = []
                wandb_images.append(wandb.Image((vis_img + 1) / 2, caption="input; prediction;"))
                wandb.log({'train-input-{}'.format(key): wandb_images}, step=epoch_1000x)
