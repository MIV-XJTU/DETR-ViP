_base_ = '../DETR-ViP_swin-t_pretrain_coco.py'
backend_args = None

model = dict(test_cfg=dict(
    max_per_img=300,
    chunked_size=40,
))

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'

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
    dict(type='MapTextToEmbedding', text_cache_file='cache/vocabulary/lvis_vocabulary.pkl'),
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

support_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='cache/support/lvis_sup.json',
        data_prefix=dict(img=data_root),
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
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_minival.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        backend_args=backend_args,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival.json')
test_evaluator = val_evaluator

val_cfg = dict(type='VisualGenericValLoop', val_mode='text')
# val_cfg = dict(type='VisualGenericValLoop', val_mode='visual-generic')

# test_cfg = dict(type='VisualGenericTestLoop', test_mode='text')
test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-generic')
# test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-interactive')