import argparse
import json
import pickle

from detr_vip.models.language_models import CLIPTextModel

def generate_od_vocabulary(args):
    json_path = args.input
    clip_path = args.clip_path
    with open(json_path, 'r') as f:
        anno = json.load(f)
    cat_name = [cat['name'] for cat in anno['categories']]

    encoder = CLIPTextModel(clip_path, 256).cuda()
    embedded = encoder(cat_name, 'cuda').detach().cpu().numpy()
    vocabulary = dict(zip(cat_name, embedded))

    save_path = args.output
    with open(save_path, 'wb') as f:
        pickle.dump(vocabulary, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('input', type=str, help='input json file name')
    parser.add_argument('--clip-path', default="path-to-clip", type=str, help='input json file name')
    parser.add_argument(
        '--output', '-o', default='cache/o365_vocabulary_.pkl', type=str, help='output json file name')
    args = parser.parse_args()

    generate_od_vocabulary(args)
