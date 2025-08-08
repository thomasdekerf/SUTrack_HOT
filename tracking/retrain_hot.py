import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description='Finetune SUTrack on HOT dataset')
    parser.add_argument('--weights', required=True, help='path to pretrained weights')
    parser.add_argument('--save_dir', required=True, help='directory to save checkpoints and logs')
    args = parser.parse_args()

    base_cfg = os.path.join(os.path.dirname(__file__), '../experiments/sutrack/sutrack_hot.yaml')
    with open(base_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('TRAIN', {})['PRETRAINED_PATH'] = args.weights

    cfg_name = 'sutrack_hot_finetune.yaml'
    cfg_path = os.path.join(os.path.dirname(base_cfg), cfg_name)
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    cmd = f"python tracking/train.py --script sutrack --config {os.path.splitext(cfg_name)[0]} --save_dir {args.save_dir}"
    os.system(cmd)


if __name__ == '__main__':
    main()
