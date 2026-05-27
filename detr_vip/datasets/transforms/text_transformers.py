# Copyright (c) MIV-XJTU. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import copy
import json
import pickle
import random
import re

import numpy as np

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

from ..utils import extract_head_noun, is_noun, get_lemma

@TRANSFORMS.register_module()
class RandomSamplingNegPosToList(BaseTransform):

    def __init__(self,
                 padding=True,
                 padding_len=80):
        self.padding = padding
        self.padding_len = padding_len

    def set_text_to_label(self, text_to_label):
        self.text_to_label = text_to_label

    def set_neg_phrase_list(self, neg_phrase_list):
        self.neg_phrase_list = neg_phrase_list

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        phrases = results['phrases']
        instance_phrases = [phrases[l]['phrase'] for l in gt_labels.tolist()]
        for i, p in enumerate(instance_phrases):
            if isinstance(p, list):
                instance_phrases[i] = ' '.join(p)
        instance_phrases = [extract_head_noun(p) for p in instance_phrases]

        is_noun_flag = [is_noun(p) for p in instance_phrases]

        instance_phrases = [get_lemma(p) for p in instance_phrases if is_noun(p)]
        gt_labels = [self.text_to_label[t] for t in instance_phrases]
        

        text = list(set(instance_phrases))
        text = sorted(text, key=lambda x:self.text_to_label[x])

        if self.padding:
            padding_len = self.padding_len - len(text)
            negative_phrases = []
            if isinstance(self.neg_phrase_list, dict):
                for p in text:
                    negative_phrases += self.neg_phrase_list[p]
                negative_phrases = list(set(negative_phrases))
                while len(negative_phrases) < padding_len:
                    p = random.choice(list(self.neg_phrase_list.keys()))
                    negative_phrases += self.neg_phrase_list[p]
                    negative_phrases = list(set(negative_phrases))
            elif isinstance(self.neg_phrase_list, list):
                negative_phrases = self.neg_phrase_list
            negative_phrases = [p for p in negative_phrases if p not in text]
            random.shuffle(negative_phrases)
            text += negative_phrases[:padding_len]
            

        text_labels = [self.text_to_label[t] for t in text]

        results['gt_bboxes'] = gt_bboxes[is_noun_flag]
        results['gt_bboxes_labels'] = np.array(gt_labels)
        results['gt_ignore_flags'] = results['gt_ignore_flags'][is_noun_flag]

        results['text'] = text
        results['text_prompt_labels'] = np.array(text_labels)
        return results

    def od_aug(self, results):
        gt_labels = results['gt_bboxes_labels']
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        
        text_labels = np.unique(gt_labels).tolist()

        label_to_text = results['text']

        text = [label_to_text[str(l)] for l in text_labels]
        label_to_positions = {l:i for i,l in enumerate(text_labels)}

        negtive_labels = [int(l) for l in label_to_text.keys() if int(l) not in text_labels]
        if self.padding:
            padding_len = self.padding_len - len(text)
            random.shuffle(negtive_labels)
            text += [label_to_text[str(l)] for l in negtive_labels[:padding_len]]
            text_labels = text_labels + negtive_labels[:padding_len]
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['text_prompt_labels'] = np.array(text_labels)

        return results

@TRANSFORMS.register_module()
class MapTextToEmbedding(BaseTransform):
    def __init__(self, text_cache_file=None):
        if text_cache_file:
            with open(text_cache_file, 'rb') as f:
                self.text_cache = pickle.load(f)
            
    def set_text_cache(self, text_cache):
        self.text_cache = text_cache

    def transform(self, results: dict) -> dict:
        if isinstance(results['text'], list) or isinstance(results['text'], tuple):
            text_prompts = [self.text_cache[text] for text in results['text']] # speaker_(stereo_equipment)
        elif isinstance(results['text'], str):
            text = results['text'].split('.')
            text = [t for t in text if t != ""]
            text_prompts = [self.text_cache[t] for t in text]
        if len(text_prompts) > 0:
            text_prompts = np.stack(text_prompts, 0) if len(text_prompts[0].shape)==1 else np.concatenate(text_prompts, 0)
        results['text_prompts'] = text_prompts
        return results