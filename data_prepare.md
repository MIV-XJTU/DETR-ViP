# Data Preparation and Processing

## DETR-ViP-T Pre-training Data Preparation and Processing

DETR-ViP-T is trained on the Objects365 V1 and GoldG datasets, and we also provide configurations trained solely on COCO and Objects365.

### 1 Objects365 v1

The corresponding training configuration is  
[DETR-ViP_swin-t_pretrain_obj365](configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py)

Objects365_v1 can be downloaded from [opendatalab](https://opendatalab.com/OpenDataLab/Objects365_v1), which supports both CLI and SDK download methods.

After downloading and decompressing, place or soft-link it under the `data/objects365v1` directory. The structure should look like:

```text
DETR-ViP
├── configs
├── data
│   ├── objects365v1
│   │   ├── objects365_train.json
│   │   ├── objects365_val.json
│   │   ├── train
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── test
```

Then convert the dataset into the ODVG format required for training using [coco2odvg.py](tools/dataset_converters/coco2odvg.py):

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
```

After the script finishes, two new files—`o365v1_train_od.json` and `o365v1_label_map.json`—will be generated under `data/objects365v1`. The complete structure becomes:

```text
DETR-ViP
├── configs
├── data
│   ├── objects365v1
│   │   ├── objects365_train.json
│   │   ├── objects365_val.json
│   │   ├── o365v1_train_od.json
│   │   ├── o365v1_label_map.json
│   │   ├── train
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── test
```

Additionally, running [prepare_OD_cache.py](tools/data_prepare/prepare_OD_cache.py) requires generating the CLIP feature cache for Objects365 category names:

```shell
python tools/data_prepare/prepare_OD_cache.py data/objects365v1/objects365_train.json --clip-path path_clip_weights --output cache/vocabulary/o365_vocabulary.pkl
```

### 2 GoldG

Once this dataset is downloaded, you can train using the configuration  
[grounding_dino_swin-t_pretrain_obj365_goldg](./grounding_dino_swin-t_pretrain_obj365_goldg.py).

The GoldG dataset includes **GQA** and **Flickr30k**, corresponding to the MixedGrounding dataset mentioned in the GLIP paper, excluding COCO. The download link is:  
[mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations).  
You will need:  
- `mdetr_annotations/final_mixed_train_no_coco.json`  
- `mdetr_annotations/final_flickr_separateGT_train.json`

Then download the [GQA images](https://nlp.stanford.edu/data/gqa/images.zip).  
After decompressing, place or soft-link them into `data/gqa`:

```text
DETR-ViP
├── configs
├── data
│   ├── gqa
│   │   ├── final_mixed_train_no_coco.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Then download the [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/).  
(Flickr30k requires application before download.)  
After decompressing, place or soft-link them into `data/flickr30k_entities`:

```text
DETR-ViP
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

For the GQA dataset, convert it to ODVG format using [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py):

```shell
python tools/dataset_converters/goldg2odvg.py data/gqa/final_mixed_train_no_coco.json
```

Afterward, `final_mixed_train_no_coco_vg.json` will be generated under `data/gqa`:

```text
DETR-ViP
├── configs
├── data
│   ├── gqa
│   │   ├── final_mixed_train_no_coco.json
│   │   ├── final_mixed_train_no_coco_vg.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

For the Flickr30k dataset, also use [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py):

```shell
python tools/dataset_converters/goldg2odvg.py data/flickr30k_entities/final_flickr_separateGT_train.json
```

Afterward, `final_flickr_separateGT_train_vg.json` will be generated under `data/flickr30k_entities`:

```text
DETR-ViP
├── configs
├── data
│   ├── flickr30k_entities
│   │   ├── final_flickr_separateGT_train.json
│   │   ├── final_flickr_separateGT_train_vg.json
│   │   ├── flickr30k_images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

<!-- Generate GoldG cache file -->
python tools/data_prepare/prepare_OG_cache.py clip_path gqa_json flickr_json save_path

### 3 COCO 2017

The above configurations evaluate on the COCO 2017 dataset during training, so COCO 2017 must also be prepared.  
You can download it from the [COCO website](https://cocodataset.org/) or from [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017).  
After decompressing, place or soft-link it under `data/coco`.

To evaluate text detection performance, you must generate the CLIP feature cache for COCO category names using [prepare_OD_cache.py](tools/data_prepare/prepare_OD_cache.py):

```shell
python tools/data_prepare/prepare_OD_cache.py data/coco/annotations/instances_train2017.json --clip-path path-to-clip --output cache/vocabulary/coco_text_cache.pkl
```

To conduct Visual-G visual prompting detection, you must sample the support set using [support_dataset.py](tools/data_prepare/support_dataset.py):

```shell
python tools/data_prepare/support_dataset.py data/coco/annotations/instances_train2017.json -o cache/support/coco_sub.json
```

## DETR-ViP-T Evaluation Data Preparation and Processing

### LVIS

To conduct Visual-G visual prompting detection, you must sample the support set using [support_dataset.py](tools/data_prepare/support_dataset.py):

```shell
python tools/data_prepare/support_dataset.py data/lvis/annotations/lvis_v1_train.json -o cache/support/lvis_sup.json --minival data/lvis/annotations/lvis_v1_minival.json
```
