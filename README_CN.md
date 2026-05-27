# [(ICLR 2026) DETR-ViP: Detection Transformer with Robust Discriminative Visual Prompts](https://arxiv.org/pdf/2604.14684)

## 简介

DETR-ViP 是一个面向**视觉提示目标检测**（Visual Prompted Object Detection）的检测框架。与传统的文本提示不同，视觉提示直接利用图像特征来定义目标类别，在识别罕见和细粒度类别上表现优异。然而，现有方法往往将视觉提示视为文本提示训练的副产品，导致其缺乏类别判别性。

DETR-ViP 通过以下三项核心创新解决了这一问题：

- **全局提示集成（Global Prompt Integration）** — 将全局类别关系融入视觉提示学习
- **视觉-文本提示关系蒸馏（Visual-Textual Prompt Relation Distillation）** — 通过知识蒸馏将判别性从文本提示迁移至视觉提示
- **选择性融合策略（Selective Fusion Strategy）** — 稳定融合视觉与文本提示，实现鲁棒检测

基于图像-文本对比学习，DETR-ViP 在 COCO、LVIS、ODinW 和 Roboflow100 上的零样本通用检测和交互式检测任务中均取得了显著提升。

本仓库包含 DETR-ViP-T 和 DETR-ViP-L 的官方实现。

![DETR-ViP 框架图](figs/framework.png)

## 实验结果

### 零样本通用检测（COCO & LVIS）

<table border="1" cellpadding="6" style="border-collapse: collapse; text-align: center; margin: 0 auto;">
  <thead>
    <tr>
      <th rowspan="2">模型</th>
      <th rowspan="2">预训练数据</th>
      <th>COCO</th>
      <th colspan="4">LVIS</th>
      <th>ODinW</th>
      <th>RF100</th>
    </tr>
    <tr>
      <th>AP</th>
      <th>AP</th>
      <th>AP<sub>f</sub></th>
      <th>AP<sub>c</sub></th>
      <th>AP<sub>r</sub></th>
      <th>AP<sub>avg</sub></th>
      <th>AP<sub>avg</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365_goldg.py">DETR-ViP-T</a></td>
      <td>O365</td>
      <td>42.3</td>
      <td>41.1</td>
      <td>40.4</td>
      <td>43.3</td>
      <td>35.1</td>
      <td>65.4</td>
      <td>66.1</td>
    </tr>
    <tr>
      <td><a href="configs/detr_vip/DETR-ViP_swin-l_pretrain_obj365_goldg.py">DETR-ViP-L</a></td>
      <td>GoldG</td>
      <td>52.4</td>
      <td>43.5</td>
      <td>42.3</td>
      <td>45.1</td>
      <td>42.9</td>
      <td>—</td>
      <td>64.2</td>
    </tr>
  </tbody>
</table>

### 零样本交互式检测（COCO & LVIS）

<table border="1" cellpadding="6" style="border-collapse: collapse; text-align: center; margin: 0 auto;">
  <thead>
    <tr>
      <th rowspan="2">模型</th>
      <th>COCO</th>
      <th colspan="4">LVIS</th>
      <th>ODinW</th>
      <th>RF100</th>
    </tr>
    <tr>
      <th>AP</th>
      <th>AP</th>
      <th>AP<sub>f</sub></th>
      <th>AP<sub>c</sub></th>
      <th>AP<sub>r</sub></th>
      <th>AP<sub>avg</sub></th>
      <th>AP<sub>avg</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365_goldg.py">DETR-ViP-T</a></td>
      <td>65.4</td>
      <td>66.1</td>
      <td>57.5</td>
      <td>73.5</td>
      <td>78.4</td>
      <td>46.8</td>
      <td>40.1</td>
    </tr>
    <tr>
      <td><a href="configs/detr_vip/DETR-ViP_swin-l_pretrain_obj365_goldg.py">DETR-ViP-L</a></td>
      <td>71.1</td>
      <td>71.9</td>
      <td>64.2</td>
      <td>78.2</td>
      <td>83.6</td>
      <td>51.2</td>
      <td>44.3</td>
    </tr>
  </tbody>
</table>

## 安装

### 依赖版本

| 包名 | 版本 |
|---------|---------|
| PyTorch | 2.0.1+cu117 |
| torchaudio | 2.0.2+cu117 |
| torchvision | 0.15.2+cu117 |
| MMCV | 2.1.0 |
| MMDetection | 3.3.0 |
| MMEngine | 0.11.0rc2 |
| numpy | 1.26.4 |
| spacy | 2.3.9 |

### MMCV

在线安装：参考 [MMCV 安装指南](https://github.com/open-mmlab/mmcv/blob/main/docs/zh_cn/get_started/installation.md)。

离线安装：
```bash
cd third_party
git clone https://github.com/open-mmlab/mmcv.git
pip install -e . -v
```

### MMDetection

在线安装：参考 [MMDetection 安装指南](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md)。

离线安装：
```bash
cd third_party
git clone https://github.com/open-mmlab/mmdetection.git
pip install -e . -v
```

### MMEngine

在线安装：参考 [MMEngine 安装指南](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/get_started/installation.md)。

离线安装：
```bash
cd third_party
git clone https://github.com/open-mmlab/mmengine.git
pip install -e . -v
```

## 数据准备

### 预训练模型

**Swin Transformer 骨干网络**（默认自动下载，离线备选）：

| 模型 | 下载链接 |
|-------|---------------|
| Swin-Tiny | [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) |
| Swin-Large | [swin_large_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth) |

若处于离线环境，请下载上述文件并放置在 `~/.cache/torch/hub/checkpoints/` 目录下。

**CLIP 模型**（用于生成类别特征缓存）：  
从 [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) 下载，并将路径设置为下文命令中的 `path_clip_weights`。

### DETR-ViP-T 预训练数据

DETR-ViP-T 在 **Objects365 V1** 和 **GoldG** 数据集上预训练，同时也提供了仅在 COCO 或 Objects365 上训练的配置。

#### 1. Objects365 V1

对应配置：[DETR-ViP_swin-t_pretrain_obj365.py](configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py)

Objects365 V1 可从 [opendatalab](https://opendatalab.com/OpenDataLab/Objects365_v1) 下载，支持 CLI 和 SDK 两种下载方式。

下载并解压后，将其放置或软链接到 `data/objects365v1` 目录下，目录结构如下：

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

使用 [coco2odvg.py](tools/dataset_converters/coco2odvg.py) 转换为 ODVG 格式：

```shell
python -m tools.dataset_converters.coco2odvg data/objects365v1/objects365_train.json -d o365v1
```

转换完成后，`data/objects365v1` 目录下会生成 `o365v1_train_od.json` 和 `o365v1_label_map.json`：

```text
DETR-ViP
├── configs
├── data
│   ├── objects365v1
│   │   ├── objects365_train.json
│   │   ├── objects365_val.json
│   │   ├── objects365_train_od.json
│   │   ├── o365v1_label_map.json
│   │   ├── train
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── test
```

生成 Objects365 类别名的 CLIP 特征缓存（[prepare_OD_cache.py](tools/data_prepare/prepare_OD_cache.py) 所需）：

```shell
python -m tools.data_prepare.prepare_OD_cache data/objects365v1/objects365_train.json --clip-path <path_clip_weights> --output cache/vocabulary/o365_vocabulary.pkl
```

#### 2. GoldG

GoldG 数据集由 **GQA** 和 **Flickr30k** 组成，源自 GLIP 论文中的 MixedGrounding 数据集（已排除 COCO）。

首先从 [mdetr_annotations](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations) 下载标注文件，需要：
- `final_mixed_train_no_coco.json`
- `final_flickr_separateGT_train.json`

**GQA 图片** 可从 [此处](https://nlp.stanford.edu/data/gqa/images.zip) 下载。下载并解压后，放置或软链接到 `data/gqa`：

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

**Flickr30k 图片** 可从 [此处](http://shannon.cs.illinois.edu/DenotationGraph/) 下载，需要先申请才能获得下载链接。下载并解压后，放置或软链接到 `data/flickr30k_entities`：

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

使用 [goldg2odvg.py](tools/dataset_converters/goldg2odvg.py) 将 GQA 标注转换为 ODVG 格式：

```shell
python -m tools.dataset_converters.goldg2odvg data/gqa/final_mixed_train_no_coco.json
```

转换完成后，`data/gqa` 目录下会生成 `final_mixed_train_no_coco_vg.json`：

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

转换 Flickr30k 标注：

```shell
python -m tools.dataset_converters.goldg2odvg data/flickr30k_entities/final_flickr_separateGT_train.json
```

转换完成后，`data/flickr30k_entities` 目录下会生成 `final_flickr_separateGT_train_vg.json`：

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

使用 [prepare_OG_cache.py](tools/data_prepare/prepare_OG_cache.py) 生成 GoldG 缓存文件：

```shell
python -m tools.data_prepare.prepare_OG_cache --clip-path <path_clip_weights> --gqa-path <gqa_json> --flickr-path <flickr_json> --save-path <save_path>
```

示例：

```shell
python -m tools.data_prepare.prepare_OG_cache --clip-path weights/clip-vit-base-patch32 --gqa-path data/GQA/final_mixed_train_no_coco_vg.json --flickr-path data/flickr/final_flickr_separateGT_train_vg.json --save-path cache/vocabulary/grounding
```

#### 3. COCO 2017

上述配置在训练过程中会评估 COCO 2017 的性能，因此需要准备该数据集。可从 [COCO 官网](https://cocodataset.org/) 或 [opendatalab](https://opendatalab.com/OpenDataLab/COCO_2017) 下载。放置或软链接到 `data/coco`。

生成 COCO 类别名的 CLIP 特征缓存（文本检测评估所需）：

```shell
python -m tools.data_prepare.prepare_OD_cache data/coco/annotations/instances_train2017.json --clip-path <path_clip_weights> --output cache/vocabulary/coco_text_cache.pkl
```

使用 [support_dataset.py](tools/data_prepare/support_dataset.py) 采样 Visual-G 视觉提示检测所需的 support 集合：

```shell
python -m tools.data_prepare.support_dataset data/coco/annotations/instances_train2017.json -o cache/support/coco_sub.json
```

### DETR-ViP-T 评测数据

> **注意：** 以下评测相关内容尚未重新验证有效性，有待后续更新。

#### LVIS

为 Visual-G 视觉提示检测采样 support 集合：

```shell
python -m tools.data_prepare.support_dataset data/lvis/annotations/lvis_v1_train.json -o cache/support/lvis_sup.json --minival data/lvis/annotations/lvis_v1_minival.json
```

## 训练

### 单卡训练

```shell
python -m tools.train <config> --work-dir <work_dirs>
```

示例：
```shell
python -m tools.train configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py --work-dir work_dirs/debug
```

### 多机多卡训练

> **注意：** 运行前请将 [tools/dist_train.sh](tools/dist_train.sh) 中的 `path-to-project` 改为你的项目根目录路径。

```shell
bash tools/dist_train.sh <config> <GPUs> --work-dir <work_dirs>
```

示例：
```shell
bash tools/dist_train.sh configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py 4 --work-dir work_dirs/debug
```

### VS Code 调试配置

配置 `.vscode/launch.json`：

```json
{
    "name": "train",
    "cwd": "${workspaceFolder}",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/tools/train.py",
    "console": "integratedTerminal",
    "env": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "args": [
        "configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365.py",
        "--work-dir", "work_dirs/debug"
    ],
    "justMyCode": false
}
```

## 评估

### 单卡评估

```shell
python -m tools.test <config> <checkpoint> --work-dir <work_dirs>
```

### 多卡评估

> **注意：** 运行前请将 [tools/dist_test.sh](tools/dist_test.sh) 中的 `path-to-project` 改为你的项目根目录路径。

```shell
bash tools/dist_test.sh <config> <checkpoint> <work_dirs> <GPUs>
```

## 模型仓库

受公司保密要求，预训练权重暂无法开放。待条件允许后会考虑开源。

## 引用

如果您的研究使用了我们的论文或代码，欢迎引用并给仓库点星。

```bibtex
@article{qian2026detr,
  title={DETR-ViP: Detection Transformer with Robust Discriminative Visual Prompts},
  author={Qian, Bo and Shi, Dahu and Wei, Xing},
  journal={arXiv preprint arXiv:2604.14684},
  year={2026}
}
```
