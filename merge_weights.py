import torch
import numpy as np
import PIL.Image
import os, sys
import re
import pickle
import logging
from typing import List, Union, Tuple
import copy

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10].
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def load_network(pickle_path):
    logging.info(f'Loading networks from "{pickle_path}"...')
    device = torch.device('cpu')
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    logging.debug(f'Network loaded successfully from {pickle_path}')
    return G, device

def log_model_parameters(G, model_name, other_model_params) -> List[Tuple[int, str, torch.Size]]:
    logging.info(f'Traversing parameters of model: {model_name}')
    param_info = []
    param_list = list(G.named_parameters())
    for idx, (name, param) in enumerate(param_list):
        logging.debug(f'Parameter: {name}, Shape: {param.shape}')
        if name not in other_model_params:
            logging.warning(f'Parameter {name} not found in the other model. Using index {idx} as fallback.')
        param_info.append((idx, name, param.shape))
    return param_info

def merge_weights_iteratively(G1, G2, iterations=10):
    for i in range(iterations):
        alpha = i / (iterations - 1)
        logging.info(f'Merging weights: iteration {i + 1}/{iterations}, alpha={alpha:.2f}')
        for (name, param_G1), (_, param_G2) in zip(G1.named_parameters(), G2.named_parameters()):
            param_G2.data = (1 - alpha) * param_G2.data + alpha * param_G1.data
        yield G2

def generate_images(G, outdir, seeds, method, iteration, truncation_psi=0.5, noise_mode='const'):
    device = next(G.parameters()).device
    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    for seed in seeds:
        logging.info(f'Generating image for seed {seed} with {method} at iteration {iteration}...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_path = os.path.join(outdir, f'{seed:04d}_{method}_iter_{iteration:02d}.jpg')
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(image_path)
        logging.debug(f'Image saved at {image_path}')

# Usage:
size = 512
project2 = "p-gen"
epoch2 = 60
project1 = "cdf"
epoch1 = 500

outdir_base = f"images/{project1}-{project2}/{size}/{epoch1}-{epoch2}"

network_pkl1 = f"networks/{project1}/{size}/network-snapshot-{epoch1:06}.pkl"
network_pkl2 = f"networks/{project2}/{size}/network-snapshot-{epoch2:06}.pkl"

G1, _ = load_network(network_pkl1)
G2, _ = load_network(network_pkl2)

G2_org = copy.deepcopy(G2)

file = os.path.basename(sys.argv[0])[:-3]
outdir = f"output_images/{file}/{project2}_{epoch2}"
seeds = [1, 42, 43]

iterations = 10  # Number of iterations for merging

for i, G2_merged in enumerate(merge_weights_iteratively(G1, G2_org, iterations)):
    generate_images(G2_merged, outdir, seeds, "merge", i)

logging.info("Completed all iterations of merging.")


