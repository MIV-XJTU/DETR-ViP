_base_ = 'DETR-ViP_swin-t_pretrain_obj365.py'

model = dict(
    bbox_head=dict(num_classes=366)
    )

o365v1_od_dataset = dict(
    type='ODVGStableDataset',
    data_root='data/objects365v1/',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    text_cache_file='cache/vocabulary/o365_vocabulary.pkl',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGStableDataset',
    data_root='data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    text_cache_file='cache/vocabulary/grounding_vocabulary.pkl',
    neg_phrase_list='cache/vocabulary/neg_list.json',
    data_prefix=dict(img='flickr30k-images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGStableDataset',
    data_root='data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    text_cache_file='cache/vocabulary/grounding_vocabulary.pkl',
    neg_phrase_list='cache/vocabulary/neg_list.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    dataset=dict(datasets=[o365v1_od_dataset, flickr30k_dataset, gqa_dataset]))
