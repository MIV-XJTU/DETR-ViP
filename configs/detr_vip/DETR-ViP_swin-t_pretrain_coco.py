_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
clip_path = "path-to-clip"
backend_args = None

model = dict(
    type='DETRViP',
    num_queries=900,
    language_dim=512,
    multiple_layer_align=True,
    training_mode='random', # random visual text
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    visual_prompt_encoder=dict(
        type="VisualPromptEncoder",
        prompt_embed_dim=256,
        num_layers=3,
        gather_prompts=True,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    encoder=dict(
        type="DETRViPTransformerEncoder",
        num_layers=6,
        num_fusion=3,
        num_cp=0, # num of checkpoint checkpoint_wrapper
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        prompt_layer_cfg=None,
        # fusion layer config
        # fusion_layer_cfg=None
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        type='DETRViPTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to prompt
            cross_attn_prompt_cfg=None,
            # cross_attn_prompt_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DETRViPHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(log_scale='auto', bias=True, memory_dim=256),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_align=dict(
            type='InfoNCELoss',
            temperature=0.1,
            loss_weight=1.0,
            normalize=False
        ),
        loss_distill=dict(
            type='RelationDistillLoss',
            temperature_v=0.1,
            temperature_t=0.07,
            # loss_weight=1.0, # ablation
            # loss_weight=10.0, # final
            loss_weight=20.0, # ablation
            normalize=True
        )),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='DETRViPBinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

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
    dict(type='RandomSamplingNegPosToList', padding=True, padding_len=80),
    dict(type='MapTextToEmbedding'),
    dict(type='RandomSamplingVisualPrompt', select_mode='training'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'text_prompt_labels', 'dataset_mode', 'text_prompts', 
                   'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MapTextToEmbedding', text_cache_file='cache/vocabulary/coco_text_cache.pkl'),
    dict(type='RandomSamplingVisualPrompt', select_mode='test'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'text_prompt_labels', 'text_prompts',
                   'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

support_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSamplingVisualPrompt', select_mode='support'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'text_prompt_labels', 'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

dataset_type = 'ODVGStableDataset'
data_root = 'data/coco/'

coco_od_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/instances_train2017_vg.json',
    label_map_file='annotations/coco2017_label_map.json',
    text_cache_file='cache/vocabulary/coco_text_cache.pkl',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[coco_od_dataset]))

support_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        # ann_file='annotations/vis_prompt_coco.json',
        ann_file='annotations/coco_subset.json',
        data_prefix=dict(img='train2017/'),
        test_mode=True,
        pipeline=support_pipeline,
        backend_args=backend_args,
        return_classes=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    support_dataloader=support_dataloader,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    # classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
    # optimizer=dict(type='AdamW', lr=0.0002,
    # optimizer=dict(type='AdamW', lr=0.0004,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

# learning policy
# max_epochs = 30
max_epochs = 12


# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[19, 26],
#         gamma=0.1)
# ]

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

train_cfg = dict(
    # type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
    type='DistributedSyncTrainLoop', max_epochs=max_epochs, val_interval=1)

# val_cfg = dict(type='VisualGenericValLoop', val_mode='text')
val_cfg = dict(type='VisualGenericValLoop', val_mode='visual-generic')

# test_cfg = dict(type='VisualGenericTestLoop', test_mode='text')
test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-generic') # text visual-generic visual-interactive
# test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-interactive')


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))
randomness = dict(seed=42)


