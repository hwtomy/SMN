import sys, os

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

import torch
torch.manual_seed(1234)

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext
import imageio
import argparse
import cv2
import gc
from lib.modelinr import ModMLP, ChebychevInput, ModMLP1, INRG, INRE, INRS, INRN
from dataio.image import ImageImplicit
from lib.util import get_coordinate_grid
from lib.op import get_nuclear_norm_regularizer
from datetime import datetime
from torchvision import transforms
from utility import recons, save_images, preimg
from skimage.metrics import peak_signal_noise_ratio as psnra
from PIL import Image
import pandas as pd

def parse_arguments():

    parser = argparse.ArgumentParser(description="Script for model training and adaptation.")

    parser.add_argument("--image_size", type=int, nargs=2, default=(768,512),
                        help="Size of the input images (width, height).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for computation (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")#For SMN, please set around 1e-3 to 2e-2.
    parser.add_argument("--pretrain_steps", type=int, default=0, help="Number of pretraining steps.")
    parser.add_argument("--adaptation_steps", type=int, default=5000, help="Number of adaptation steps.")
    parser.add_argument("--n_pretrain", type=int, default=0, help="Number of pretraining steps.")
    parser.add_argument("--n_adaptation", type=int, default=24, help="Number of adaptation steps.")#This item is indeed the number of images in the target dataset
    parser.add_argument("--n_samples_per_image", type=int, default=393216, help="Number of samples per image.")
    parser.add_argument("--hidden_features", type=int, default=256, help="Number of hidden features in the model.")
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers in the model.")
    parser.add_argument("--lora_rank", type=int, nargs='+', default=0, help="Rank for LoRA layers.")
    parser.add_argument("--inr_type", type=str, default="siren", help="INR type.")#For SMN, please set to "siren".
    parser.add_argument("--mod_type", type=str, default="lora_vanilla", help="modulation type.")
    parser.add_argument("--pretrain_with_mod", action='store_true', help="Pretrain with modulation.")
    parser.add_argument("--mod_with_bias", action='store_true', help="Modulate with bias.")
    parser.add_argument("--weight_norm", action='store_true', help="Use weight normalization.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--data_dir", type=str, default="div2k/",
                        help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="outputTest/",
                        help="Path to the dataset directory.")
    parser.add_argument("--nuclear_norm_weight", type=float, default=0.0, help="Weight for nuclear norm regularization.")
    args = parser.parse_args()

    if isinstance(args.lora_rank, list) and len(args.lora_rank) == 1:
        args.lora_rank = args.lora_rank[0]

    return args


def run_optimization(model, dataloader,  optim, steps, min_lr=1e-5, use_weight_norm=False, use_amp=False, nuclear_norm_weight=0.0):

    scaler = GradScaler() if use_amp else None
    autocast_context = autocast() if use_amp else nullcontext()
    tbar = tqdm(range(steps))

    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=100, min_lr=min_lr)

    data_iter = iter(dataloader)

    for epoch in tbar:

        data = next(data_iter)
        coords = data["query"]
        gt = data["gt"]
        size = data["size"]

        with autocast_context:
            outputs = model(coords)
            # image = outputs
            outputs = torch.sigmoid(outputs)
            gt = gt * 0.5 + 0.5

            loss = torch.nn.functional.mse_loss(outputs, gt, reduction="none")
            loss = loss.mean(dim=[1, 2])

            psnr = -10 * torch.log10(loss)
            psnr = psnr.mean()

            loss = loss.sum()

            if nuclear_norm_weight > 0:
                nuclear_norm = get_nuclear_norm_regularizer(mod_params)
                if torch.is_tensor(nuclear_norm):
                    loss = loss + nuclear_norm_weight * nuclear_norm.sum()

        optim.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        else:
            loss.backward()
            optim.step()

        if use_weight_norm:
            model.apply_weight_norm(mod_params)

        scheduler.step(loss.item(), epoch=epoch)

        current_lr = [param_group['lr'] for param_group in optim.param_groups][0]
        tbar.set_description(f"Iter {epoch}/{steps} LR = {current_lr:.2e} Loss = {loss.item():.6f} PSNR = {psnr.item():.2f}")
        tbar.refresh()

    return psnr.item()

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def main(opt):
    torch.cuda.empty_cache()
    files = os.listdir(opt.data_dir)
    files = sorted(files)
    os.makedirs(opt.output_dir, exist_ok=True)




    # ADAPTATION
       
    model = ModMLP(2, 3, opt.hidden_features, n_hidden_layers=opt.n_hidden_layers,
                    inr_type=opt.inr_type, mod_type=opt.mod_type, skip_mode="cat").to(opt.device)
    # model = ModMLP1(2, 3, hidden_features=300,  n_hidden_layers=3,
    #                 inr_type="rinr", mod_type=opt.mod_type, skip_mode="cat").to(opt.device)

    # model = INRG(in_features=2,
    #              hidden_features=420,
    #              hidden_layers=3,
    #              out_features=3,
    #              outermost_linear=True,
    #              first_omega_0=20.0,
    #              hidden_omega_0=1.0,
    #              scale=10.0).to(opt.device)
    if not opt.n_adaptation > 0: return
    all_psnrs = []
    for i in range(opt.n_adaptation):
        # if 'model' in locals():
        #     del model
        
        torch.cuda.empty_cache()
        gc.collect()
        model.apply(reset_weights)
        image_path = os.path.join(opt.data_dir, files[i])
        print(f"Processing image: {files[i]}")
        adaptation_dl = ImageImplicit([image_path], opt.image_size, opt.n_samples_per_image, device=opt.device)

        n_params_base = sum([v.numel() for v in model.parameters()])
        print(f"# base model params: {n_params_base}")
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))#2e-2
        psnr = run_optimization(model, adaptation_dl,  optim, opt.adaptation_steps, use_weight_norm=opt.weight_norm, use_amp=False, nuclear_norm_weight=opt.nuclear_norm_weight)
        all_psnrs.append(psnr)


        # INFERENCE
        os.makedirs(opt.output_dir, exist_ok=True)
        for j in range(opt.n_adaptation):
            with torch.no_grad():
                query_coordinates = get_coordinate_grid(1, opt.image_size, device=opt.device)
   
                outputs = model(query_coordinates)
                outputs = outputs.cpu()

                output = torch.clamp(outputs[0].permute(1, 2, 0), -1, 1) * 0.5 + 0.5
                output = torch.sigmoid(outputs[0].permute(1, 2, 0))
                output = (output * 255).to(torch.uint8).numpy()

                # print(output.shape)
                imageio.imwrite(f"{opt.output_dir}/try_loramod_adaptation_{i:03d}.png", output)

            # torch.save({k: (v[i] if len(v) > 0 else v) for k, v in mod_params.items() if not "lora_alpha" in k},
            #            os.path.join(opt.output_dir, f"mod_params_{i}.pth"))

    print(all_psnrs)
    df = pd.DataFrame([round(v, 2) for v in all_psnrs], columns=["PSNR"])


    output_path = "psnrvaluesMLP.xlsx"
    df.to_excel(output_path, index=False)
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    main(args)