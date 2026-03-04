import json
import pickle
import spacy
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F

from detr_vip.models.language_models import CLIPTextModel
from mmengine.fileio import get_local_path

from detr_vip.datasets.utils import extract_head_noun, is_noun, get_lemma

@torch.no_grad()
def generate_vocabulary(clip_path, gqa_path, flickr_path, save_path):
    encoder = CLIPTextModel(clip_path, 256).cuda()
    with get_local_path(
            gqa_path, backend_args=None) as local_path:
        with open(local_path, 'r') as f:
            data_list = [json.loads(line) for line in f]
            
    with get_local_path(
            flickr_path, backend_args=None) as local_path:
        with open(local_path, 'r') as f:
            data_list += [json.loads(line) for line in f]


    phrase_items = {}
    _cnt = 0
    for data in tqdm(data_list):
        # if _cnt > 100:
        #     break
        _cnt += 1
        regions = data['grounding']['regions']
        phrase = [r['phrase'] for r in regions]
        for p in phrase:
            if isinstance(p, list):
                p = ' '.join(p)
            p = extract_head_noun(p)
            if not is_noun(p):
                continue
            p = get_lemma(p)
            if p in phrase_items:
                phrase_items[p]['fre'] += 1
            else:
                phrase_items[p] = {}
                # phrase_items[p]['embedded'] = encoder([p], 'cuda').detach().cpu().numpy()
                phrase_items[p]['fre'] = 1
    
    neg_phrase_list = []
    for p, t in phrase_items.items():
        if t['fre'] > 100:
            neg_phrase_list.append(p)

    phrase_list = list(phrase_items.keys())[:100]
    embedded = encoder(phrase_list, 'cuda').detach().cpu().numpy()
    phrase_items = dict(zip(phrase_list, embedded))
    neg_list_save_path = save_path+"_neg_list.json"
    vocabulary_save_path = save_path+"_vocabulary.pkl"
    with open(neg_list_save_path, 'w') as f:
        json.dump(neg_phrase_list, f, indent=4)
    with open(vocabulary_save_path, 'wb') as f:
        pickle.dump(phrase_items, f)

    
if __name__ == "__main__":
    clip_path = "path-to-clip"
    gqa_path = "path-to-gqa-json"
    flickr_path = "path-to-flickr-json"
    save_path = "save-path"
    generate_vocabulary(clip_path, gqa_path, flickr_path, save_path)

    