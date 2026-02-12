# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

import argparse
from model.gpt import GPT_models
from language.t5 import T5Embedder
from model.generate import generate
import numpy as np
import json
import os
import pickle

from config import get_decoding_config


def decoding(index_sample, config=None):
    """
    Decode token indices to JSON format building descriptions.
    
    Args:
        index_sample: Tensor of token indices from the model
        config: DecodingConfig instance (optional, uses default if None)
    
    Returns:
        List of JSON-serializable dictionaries with building elements
    """
    if config is None:
        config = get_decoding_config()
    
    with open("./data/class_to_id.pkl", "rb") as f:
        class_to_id = pickle.load(f)
    
    # Get coordinate mappings from config
    xclass_x, yclass_y, zclass_z = config.get_coordinate_maps()
    
    # Get token range offsets from config
    x_offset = config.token_ranges.location_x['start']
    y_offset = config.token_ranges.location_y['start']
    z_offset = config.token_ranges.location_z['start']
    yaw_offset = config.token_ranges.rotation_yaw['start']

    json_list = []

    for sample_id in range(index_sample.shape[0]):
        sample = index_sample[sample_id]
        try:
            eos_pos = np.argwhere(sample.cpu().numpy() == config.eos_id)[0][0]
        except IndexError:
            eos_pos = len(sample)

        json_file = {}
        json_file['elements'] = []

        type_index = 1
        for token_id in range(eos_pos):
            token = sample[token_id].float()

            if type_index == 1:
                building_id = int(token)
                building_class = class_to_id[building_id]

            elif type_index == 2:
                location_x = xclass_x[int(token) - x_offset]
            elif type_index == 3:
                location_y = yclass_y[int(token) - y_offset]
            elif type_index == 4:
                location_z = zclass_z[int(token) - z_offset]
            elif type_index == 5:
                rotation_yaw = (int(token) - yaw_offset) * config.yaw_step + config.yaw_base
                json_file['elements'].append({
                    "building_id": building_class, 
                    "location": {"x": location_x, "y": location_y, "z": location_z}, 
                    "rotation": {"roll": 0, "pitch": 0, "yaw": rotation_yaw}, 
                    "floor": 0
                })
                type_index = 0
            
            type_index += 1
            
        json_list.append(json_file)
    return json_list


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    
    # Check if gpt_ckpt is provided
    if args.gpt_ckpt is None:
        raise ValueError("--gpt-ckpt is required. Please provide a path to the model checkpoint.")
    
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        seq_len=5550,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )
    else:
        print(f"no need to compile model in demo") 

    save_path = "/".join(args.gpt_ckpt.split("/")[:-2]) + '/samples/'
    os.makedirs(save_path, exist_ok=True)
    
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    # support only structured prompts
    # check the prompt format in Fig. 14 and Tab. 7. of the paper
    prompts = [
        "apartment, multi-story, pitched roof, minimal windows",
        "mansion, multi-story, flat roof, lots of windows",
        "courtyard, single-story, pitched roof, few windows",
        "castle, single-story, flat roof, simple design"
    ]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)
    t5_feature_max_len = args.cls_token_num
    caption_embs = caption_embs[:,:t5_feature_max_len]
    emb_masks = emb_masks[:,:t5_feature_max_len]
    new_emb_masks = torch.flip(emb_masks, dims=[-1])
    new_caption_embs = []
    for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
        valid_num = int(emb_mask.sum().item())
        new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
        new_caption_embs.append(new_caption_emb)
    new_caption_embs = torch.stack(new_caption_embs)
    
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks
    
    index_sample = generate(
        gpt_model, c_indices, 5550, c_emb_masks,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )

    samples = decoding(index_sample)

    for idx, json_file in enumerate(samples):
        with open(save_path+f"sample_{prompts[idx]}.json", "w") as f:
            json.dump(json_file, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=24, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--codebook-size", type=int, default=214, help="codebook size for vector quantization")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)