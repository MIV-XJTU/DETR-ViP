_base_ = 'DETR-ViP_swin-l_pretrain_coco.py'
backend_args = None
model = dict(
    bbox_head=dict(num_classes=365)
    )

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPosToList', padding=True, padding_len=365),
    dict(type='MapTextToEmbedding'),
    dict(type='RandomSamplingVisualPrompt', select_mode='training'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'text_prompt_labels', 'dataset_mode', 'text_prompts', 
                   'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

o365v1_od_dataset = dict(
    type='ODVGStableDataset',
    data_root='data/objects365v1/',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    text_cache_file='cache/vocabulary/o365_vocabulary.pkl',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None,
)

train_dataloader = dict(
    dataset=dict(datasets=[o365v1_od_dataset]))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002,
    # optimizer=dict(type='AdamW', lr=0.0004,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))
