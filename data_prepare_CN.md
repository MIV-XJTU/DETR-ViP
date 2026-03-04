# 数据准备和处理

## DETR-ViP-T 预训练数据准备和处理

DETR-ViP-T在Objects365 V1和GoldG数据集上进行训练，同时提供了在COCO、Objects365上单独训练的配置。

### 1 Objects365 v1

对应的训练配置为 [DETR-ViP_swin-t_pretrain_obj365](configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py)

Objects365_v1 可以从 [opendatalab](https://opendatalab.com/OpenDataLab/Objects365_v1) 下载，其提供了 CLI 和 SDK 两者下载方式。

下载并解压后，将其放置或者软链接到 `data/objects365v1` 目录下，目录结构如下：

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

然后使用 [coco2odvg.py](tools/dataset_converters/coco2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
```

程序运行完成后会在 `data/objects365v1` 目录下创建 `o365v1_train_od.json` 和 `o365v1_label_map.json` 两个新文件，完整结构如下：

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

此外运行[prepare_OD_cache.py](tools/data_prepare/prepare_OD_cache.py)还需要提前生成Objects365的类别名的CLIP Feature Cache：
```shell
python tools/data_prepare/prepare_OD_cache.py data/objects365v1/objects365_train.json --clip-path path_clip_weights --output cache/vocabulary/o365_vocabulary.pkl
```

### 2 GoldG

下载该数据集后就可以训练 [grounding_dino_swin-t_pretrain_obj365_goldg](./grounding_dino_swin-t_pretrain_obj365_goldg.py) 配置了。

GoldG 数据集包括 `GQA` 和 `Flickr30k` 两个数据集，来自 GLIP 论文中提到的 MixedGrounding 数据集，其排除了 COCO 数据集。下载链接为 [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations)，我们目前需要的是 `mdetr_annotations/final_mixed_train_no_coco.json` 和 `mdetr_annotations/final_flickr_separateGT_train.json` 文件。

然后下载 [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) 图片。下载并解压后，将其放置或者软链接到 `data/gqa` 目录下，目录结构如下：

```text
DETR-ViP
├── configs
├── data
│   ├── gqa
|   |   ├── final_mixed_train_no_coco.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

然后下载 [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/) 图片。这个数据下载需要先申请，再获得下载链接后才可以下载。下载并解压后，将其放置或者软链接到 `data/flickr30k_entities` 目录下，目录结构如下：

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

对于 GQA 数据集，你需要使用 [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/goldg2odvg.py data/gqa/final_mixed_train_no_coco.json
```

程序运行完成后会在 `data/gqa` 目录下创建 `final_mixed_train_no_coco_vg.json` 新文件，完整结构如下：

```text
DETR-ViP
├── configs
├── data
│   ├── gqa
|   |   ├── final_mixed_train_no_coco.json
|   |   ├── final_mixed_train_no_coco_vg.json
│   │   ├── images
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

对于 Flickr30k 数据集，你需要使用 [goldg2odvg.py](../../tools/dataset_converters/goldg2odvg.py) 转换为训练所需的 ODVG 格式：

```shell
python tools/dataset_converters/goldg2odvg.py data/flickr30k_entities/final_flickr_separateGT_train.json
```

程序运行完成后会在 `data/flickr30k_entities` 目录下创建 `final_flickr_separateGT_train_vg.json` 新文件，完整结构如下：

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

<!-- 获取GlodG cache文件 -->
python tools/data_prepare/prepare_OG_cache.py clip_path gqa_json flikcr_json save_path

### 3 COCO 2017
上述配置在训练过程中会评估 COCO 2017 数据集的性能，因此需要准备 COCO 2017 数据集。你可以从 [COCO](https://cocodataset.org/) 官网下载或者从 [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017) 下载。
下载并解压后，将其放置或者软链接到 `data/coco` 目录下。

若要测试文本检测的性能，则需提前运行[prepare_OD_cache.py](tools/data_prepare/prepare_OD_cache.py)还需要提前生成COCO的类别名的CLIP Feature Cache：
```shell
python tools/data_prepare/prepare_OD_cache.py data/coco/annotations/instances_train2017.json --clip-path path-to-clip --output cache/vocabulary/coco_text_cache.pkl
```

进行Visual-G的视觉提示检测需要提前运行[support_dataset.py](tools/data_prepare/support_dataset.py)采样support集合：
```shell
python tools/data_prepare/support_dataset.py data/coco/annotations/instances_train2017.json -o cache/support/coco_sub.json
```

## DETR-ViP-T 评测数据准备和处理
### LVIS

进行Visual-G的视觉提示检测需要提前运行[support_dataset.py](tools/data_prepare/support_dataset.py)采样support集合：
```shell
python tools/data_prepare/support_dataset.py data/lvis/annotations/lvis_v1_train.json -o cache/support/lvis_sup.json --minival data/lvis/annotations/lvis_v1_minival.json
```