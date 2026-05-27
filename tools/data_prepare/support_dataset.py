import argparse
import json
import random

from pycocotools.coco import COCO

def generate_supset(args):
    ann_file = args.input
    save_path = args.output
    minival_path = args.minival
    valid_cls_id = []
    
    if minival_path is not None:
        lvis = COCO(minival_path)
        for m_anno in lvis.dataset["annotations"]:
            valid_cls_id.append(m_anno['category_id'])
        valid_cls_id = list(set(valid_cls_id))
    coco = COCO(ann_file)

    categories = coco.loadCats(coco.getCatIds()) 
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories} 
    cat_ids = list(cat_id_to_name.keys())  

    N = 16
    selected_img_ids = set()

    imgid2catid = {}
    image_ids = coco.getImgIds()
    for img_id in image_ids:
        imgid2catid[img_id] = []
    for cat_id in cat_ids:
        if len(valid_cls_id)> 0 and cat_id not in valid_cls_id:
            continue
        img_ids = coco.getImgIds(catIds=[cat_id])
        selected_ids = random.sample(img_ids, min(N, len(img_ids)))
        for selected_id in selected_ids:
            imgid2catid[selected_id].append(cat_id)
        selected_img_ids.update(selected_ids)

    for i in selected_img_ids:
        print(imgid2catid[i])

    selected_img_ids = list(selected_img_ids) 
    selected_images = coco.loadImgs(selected_img_ids)  

    selected_annotations = [ann for ann in coco.dataset["annotations"] if ann["image_id"] in selected_img_ids and ann['category_id'] in imgid2catid[ann["image_id"]]]

    new_images = []
    image_id_map = {} 
    new_img_id = 1

    for img in selected_images:
        image_id_map[img["id"]] = new_img_id 
        img["id"] = new_img_id 
        new_images.append(img)
        new_img_id += 1

    new_annotations = []
    new_ann_id = 1

    for ann in selected_annotations:
        ann["image_id"] = image_id_map[ann["image_id"]]
        ann["id"] = new_ann_id
        new_annotations.append(ann)
        new_ann_id += 1

    # 构造新的 COCO JSON
    new_coco_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories
    }

    with open(save_path, "w") as f:
        json.dump(new_coco_data, f, indent=4)

    print(f"新的 COCO JSON 子集已保存，共选取 {len(selected_img_ids)} 张图片！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('input', type=str, help='input json file name')
    parser.add_argument(
        '--output', '-o', default='lvis_sub.json', type=str, help='output json file name')
    parser.add_argument(
        '--minival', default='data/lvis/annotations/lvis_v1_minival.json', type=str, help='minival json')
    args = parser.parse_args()
    

    generate_supset(args)