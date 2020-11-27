import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from configs.default import cfg
from engine.evaluator import Evaluator as e
from utils.general import set_gpu

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', default='')
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    args = parser.parse_args()

    if args.cfg:
        cfg.merge_from_file(args.cfg)

    set_gpu(args.device)
    evaluator = e(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()
