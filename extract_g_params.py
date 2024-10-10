"""
extract_g_params.py

A script for extracting the learned Gaussian parameters of SPAGHETTI,
which will be used as training data for the diffusion model.
"""

from datetime import datetime
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

import constants
from options import Options
# from train import TrainerInOut
from utils import train_utils, files_utils


def init_model(opt: Options):
    """
    Initializes model and loads the checkpoint.

    Returns:
    - the model initialized using the given configuration and with weights loaded;
    - the configuration used for initialization.
    """
    print(f"[!] Checkpoint folder: {opt.cp_folder}")

    model_, opt_ = train_utils.model_lc(opt)
    model_.eval()
    print(f"[!] Initialized model.")
    return model_, opt_


@torch.no_grad()
def extract_data(model, data):
    # Sample z^a's
    za = model.get_z(data)
    print(f"[!] Sampled {za.shape[0]} z^a's.")    
    # Decomposition network: z^a -> Z^b
    Zb = model.decomposition_control.forward_low(za)
    # Zb, _ = model.forward_low(za)
    # print(f"[!] ====>Decomposed {Zb.shape[0]} z^a's.")
    # exit()
    # Decomposition network: Affine-transformed GMMs
    s_j, g_js = model.decomposition_control.forward_mid(Zb)  # Z^b -> (s_j, g_j)
    # Decomposition network: Non-transformed GMMs
    reflect_backup = model.decomposition_control.reflect
    if reflect_backup is None:
        print("[!] 'OccFormer' reflect is initiall None.")
    model.decomposition_control.reflect = None
    s_j_, g_js_ = model.decomposition_control.forward_mid(Zb)  # Z^b -> (s_j, g_j)
    model.decomposition_control.reflect = reflect_backup
    assert torch.equal(s_j, s_j_), "[!] Intrinsic latents are irrelevant to Affine transformations."
    # Combine s_j's and g_j's
    Zb_hat = model.merge_zh_step_a(s_j, g_js)  # w/ Affine
    Zb_hat_ = model.merge_zh_step_a(s_j, g_js_)  # w/o Affine
    # Mixing network: Z^c
    # Note that we do not mask any part embedding during inference
    Zc, attn = model.mixing_network.forward_with_attention(Zb_hat, mask=None)  # w/ Affine
    Zc_, attn = model.mixing_network.forward_with_attention(Zb_hat_, mask=None)  # w/o Affine

    return za, Zb, (s_j, s_j_), (g_js, g_js_), (Zb_hat, Zb_hat_), (Zc, Zc_)
def parse_gmms(gmms):
    assert len(gmms) == 1, f"[!] Length of GMMs is {len(gmms)}, exceeds expected length 1."
    if isinstance(gmms, list):
        assert len(gmms)

    mu = gmms[0][0]  # (B, 1, G, 3)
    ps = gmms[0][1]  # (B, 1, G, 3, 3)
    phi = gmms[0][2]  # (B, 1, G)
    eigen = gmms[0][3]  # (B, 1, G, 3)

    B, _, G, _ = mu.shape
    ps = ps.reshape(B, 1, G, -1)  # (B, 1, G, 3, 3) -> (B, 1, G, 9)
    phi = phi[..., None]  # (B, 1, G) -> (B, 1, G, 1)

    concat = torch.cat([mu, ps, phi, eigen], -1)
    concat = concat.reshape(B, G, -1)  # (B, G, 16)
    assert concat.shape == (B, G, 16)

    return concat
def main(opt):
    # Initialize the model.
    model, opt = init_model(opt)
    opt.batch_size = 1
    num_samples = model.z.weight.shape[0]
    print(f"[!] Number of pretrained embeddings: {num_samples}")
    print(f"[!] Batch size: {opt.batch_size}")
    print("batch_size:", opt.batch_size)
    # exit()
    indices = torch.arange(0, num_samples).long().to(model.device)
    all_za = []
    all_Zb = []
    all_s_j_affine = []
    all_s_j = []
    all_g_js_affine = []
    all_g_js = []
    all_Zb_hat_affine = []
    all_Zb_hat = []
    all_Zc_affine = []
    all_Zc = []
    for batch in tqdm(indices.split(opt.batch_size)):
        # Extract SPAGHETTI latents
        (
            za,
            Zb,
            (s_j_affine, s_j),
            (g_js_affine, g_js),
            (Zb_hat_affine, Zb_hat),
            (Zc_affine, Zc),
        ) = extract_data(model, batch)

        # Checks on data
        print("see len of Zb:", len(Zb))
        print("see opt.batch_size:", opt.batch_size)

        assert len(Zb) ==  1
        Zb = Zb[-1]  # [ ... ] -> (B, G, 512)
        print("see Zb shape:", Zb.shape)#  torch.Size([16, 512])
        # exit()
        assert (
            len(g_js_affine) == 1
        ), f"[!] Length of GMMs is {len(g_js_affine)}, exceeds expected length 1."
        assert len(g_js) == 1, f"[!] Length of GMMs is {len(g_js)}, exceeds expected length 1."

        # Collect 'za' and 'Zb'
        all_za.append(za)
        all_Zb.append(Zb)

        # Collect 's_j's
        all_s_j_affine.append(s_j_affine)
        all_s_j.append(s_j)

        # Collect GMM (extrinsic) parameters
        all_g_js_affine.append(parse_gmms(g_js_affine))
        all_g_js.append(parse_gmms(g_js))

        # Collect 'Zb_hat's
        all_Zb_hat_affine.append(Zb_hat_affine)
        all_Zb_hat.append(Zb_hat)

        # Collect 'Zc's
        all_Zc_affine.append(Zc_affine)
        all_Zc.append(Zc)
    data_to_save = {
        "za": torch.cat(all_za, 0).cpu().numpy(),
        "Zb": torch.cat(all_Zb, 0).cpu().numpy(),
        "s_j_affine": torch.cat(all_s_j_affine, 0).cpu().numpy(),
        "s_j": torch.cat(all_s_j, 0).cpu().numpy(),
        "g_js_affine": torch.cat(all_g_js_affine, 0).cpu().numpy(),
        "g_js": torch.cat(all_g_js, 0).cpu().numpy(),
        "Zb_hat_affine": torch.cat(all_Zb_hat_affine, 0).cpu().numpy(),
        "Zb_hat": torch.cat(all_Zb_hat, 0).cpu().numpy(),
        "Zc_affine": torch.cat(all_Zc_affine, 0).cpu().numpy(),
        "Zc": torch.cat(all_Zc, 0).cpu().numpy(),
    }

    # Permute the samples to match the ordering.
    with open(
        "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/spaghetti_full_shapenet_sorted_sn.txt"
    ) as f:
        spaghetti_sorted_sn = [line.rstrip() for line in f]
    with open("/home/juil/docker_home/projects/3D_CRISPR/crispr/data/sorted_sn.txt") as f:
        partglot_sorted_sn = [line.rstrip() for line in f]
    assert len(spaghetti_sorted_sn) == len(data_to_save["za"])

    indices_for_shuffle = []
    for par_sn in partglot_sorted_sn:
        for spa_idx, spa_sn in enumerate(spaghetti_sorted_sn):
            if spa_sn == par_sn:
                indices_for_shuffle.append(spa_idx)
                break
    assert len(indices_for_shuffle) == len(partglot_sorted_sn)
    num_samples_shuffled = len(indices_for_shuffle)  # Update the number of samples

    # Log data into a HDF5 file.
    outpath = os.path.join(constants.DATA_ROOT, "latent_params")
    os.makedirs(outpath, exist_ok=True)
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    hdf5 = os.path.join(outpath, f"{num_samples}_{time_str}.hdf5")
    hdf5_shuffled = os.path.join(outpath, f"{num_samples_shuffled}_{time_str}.hdf5")

    with h5py.File(hdf5, "w") as f:
        for d_name, data in data_to_save.items():
            _ = f.create_dataset(d_name, data=data)
    with h5py.File(hdf5_shuffled, "w") as f:
        for d_name, data in data_to_save.items():
            data = data[indices_for_shuffle]
            _ = f.create_dataset(d_name, data=data)

    # Sanity check. Two arrays must be equivalent.
    f_ = h5py.File(hdf5_shuffled, "r")
    for d_name, data in data_to_save.items():
        assert np.array_equal(f_[d_name], data[indices_for_shuffle])
    f_ = h5py.File(hdf5, "r")
    for d_name, data in data_to_save.items():
        assert np.array_equal(f_[d_name], data)
    print(f"[!] Successfully saved file at {hdf5}")
    print(f"[!] Successfully saved file at {hdf5_shuffled}")


if __name__ == "__main__":
    opt = Options()
    
    main(opt)
