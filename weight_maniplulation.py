import torch
import numpy as np
import PIL.Image
import os, sys
import re
import pickle
import logging
from typing import List, Union, Tuple
import copy  # Import copy module for deepcopy

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
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

# Load the generator from the given pickle file using native Torch functions
def load_network(pickle_path):
    logging.info(f'Loading networks from "{pickle_path}"...')
    device = torch.device('cuda')  # Use GPU
    #device = torch.device('cpu')  # Use CPU
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # Load the generator and move to GPU
    logging.debug(f'Network loaded successfully from {pickle_path}')
    return G, device

# Function to log model parameters and check existence in the other model
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

# Normalization function
def normalize_weight(weight):
    return (weight - weight.mean()) / (weight.std() + 1e-5)

# Scaling function
def scale_weight(weight, factor=0.5):
    return weight * factor

# Excitation function
def excite_weight(weight, factor=2):
    return torch.tanh(weight) * factor

# Calming function
def calm_weight(weight, factor=0.5):
    return torch.sigmoid(weight) * factor

# Function to merge or replace weights
def manipulate_weights(G1, G2, method='add', weight_key=None, index=None) -> Tuple[torch.nn.Module, bool]:
    try:
        if weight_key:
            logging.debug(f'Accessing parameter by key: {weight_key}')
            original_weight = dict(G2.named_parameters())[weight_key]
            manipulation_weight = dict(G1.named_parameters())[weight_key]
        elif index is not None:
            logging.debug(f'Accessing parameter by index: {index}')
            original_weight = list(G2.parameters())[index]
            manipulation_weight = list(G1.parameters())[index]
            weight_key = list(G1.named_parameters())[index][0]
        else:
            raise ValueError("Either weight_key or index must be provided")

        logging.debug(f'Original weight shape: {original_weight.shape}')
        logging.debug(f'Manipulation weight shape: {manipulation_weight.shape}')

        # Check if the shapes are compatible
        if original_weight.shape != manipulation_weight.shape:
            logging.warning(f'Skipping weight {weight_key} due to shape mismatch: '
                            f'{original_weight.shape} vs {manipulation_weight.shape}')
            return G2, False

        # Manipulate weights based on the method
        if method == 'add':
            new_weight = original_weight + manipulation_weight
            logging.info(f'Added weights: {weight_key}')
        elif method == 'replace':
            new_weight = manipulation_weight
            logging.info(f'Replaced weight: {weight_key}')
        elif method == 'normalize':
            new_weight = normalize_weight(original_weight)
            logging.info(f'Normalized weight: {weight_key}')
        elif method == 'scale':
            new_weight = scale_weight(original_weight)
            logging.info(f'Scaled weight: {weight_key}')
        elif method == 'excite':
            new_weight = excite_weight(original_weight)
            logging.info(f'Excited weight: {weight_key}')
        elif method == 'calm':
            new_weight = calm_weight(original_weight)
            logging.info(f'Calmed weight: {weight_key}')
        else:
            raise ValueError("Unsupported manipulation method")

        # Assign the new weight to G2
        original_weight.data.copy_(new_weight)
        return G2, True

    except KeyError as e:
        logging.warning(f'Skipping weight {weight_key} due to missing key: {e}')
        return G2, False

# Main function to generate images
def generate_images(G, outdir, seeds, method, weight_key, truncation_psi=0.5, noise_mode='const'):
    device = next(G.parameters()).device
    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    for seed in seeds:
        logging.info(f'Generating image for seed {seed} with {method} on {weight_key}...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_path = os.path.join(outdir, f'{seed:04d}_{method}_{weight_key.replace(".", "_")}.jpg')
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(image_path)
        logging.debug(f'Image saved at {image_path}')

# Usage:
size = 1024
project1 = "ca"
epoch1 = 2300
project2 = "ad"
epoch2 = 2400

outdir = f"images/{project1}-{project2}/{size}/{epoch1}-{epoch2}"  # Output directory for generated images

network_pkl1 = f"networks/{project1}/{size}/network-snapshot-{epoch1:06}.pkl"  # Path to your pickle file
network_pkl2 = f"networks/{project2}/{size}/network-snapshot-{epoch2:06}.pkl"  # Path to your pickle file

G1, _ = load_network(network_pkl1)
G2, _ = load_network(network_pkl2)

G2_org = copy.deepcopy(G2)  # Create a deep copy of G2

# Get the parameter information from both models
G1_params = {name: param.shape for name, param in G1.named_parameters()}
G2_params_info = log_model_parameters(G2, 'Model 2', G1_params)

file = os.path.basename(sys.argv[0])[:-3]
outdir_base = f"output_images/{file}/{project2}_{epoch2}"
seeds = parse_range("1-20")

# Loop through weights and manipulation methods
methods = ['add', 'replace', 'normalize', 'scale', 'excite', 'calm']
for idx, (name, param) in enumerate(G1.named_parameters()):
    for method in methods:
        logging.info(f'Starting {method} manipulation for weight: {name}')
        # Use a fresh copy of G2_org for each manipulation to avoid progressive changes
        G2_copy = copy.deepcopy(G2_org)
        G2, success = manipulate_weights(G1, G2_copy, method=method, weight_key=name, index=idx)
        if success:
            #outdir = f"{outdir_base}/{method}/{name.replace('.', '_')}"
            outdir = f"{outdir_base}/{method}"
            generate_images(G2, outdir, seeds, method, name)
        else:
            logging.info(f'Skipping image generation for {method} on {name} due to unsuccessful manipulation.')

logging.info("Completed all variations.")

