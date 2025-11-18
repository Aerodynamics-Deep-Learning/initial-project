from config_defs import cfg_from_args, parse_cfg_dict
from data_loaders import *
from loaders import *
from train import train
from auxiliary import setup_wandb

import traceback

if __name__ == "__main__":

    cfg_run = cfg_from_args()

    cfg_data, cfg_loader, cfg_train, cfg_export, cfg_model_setup, cfg_optim_setup, cfg_loss_setup = parse_cfg_dict(cfg_run=cfg_run)

    # Setup wandb for logging
    run = setup_wandb(cfg_run= cfg_run, cfg_export= cfg_export)

    print("WandB has been setup.")

    # Initialize Data, Model, Optim, Loss
    dataloaders = create_dataloaders(cfg_data= cfg_data, cfg_loader= cfg_loader)
    model = load_model(cfg_model_setup= cfg_model_setup)
    optimizer, scheduler = load_optim(cfg_optim_setup= cfg_optim_setup, model= model)
    loss_fn = load_loss(cfg_loss_setup= cfg_loss_setup)

    try:
        train(cfg_train=cfg_train, cfg_export=cfg_export, model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, dataloaders=dataloaders)

    except Exception as e:
        run.alert(title= "Training Crashed", text=str(e))
        traceback.print_exc()
        raise e

    finally:
        run.finish()








