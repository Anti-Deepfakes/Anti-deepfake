import os
import argparse
import torch
from utils.train_v1 import train
from utils.dataloader_v1 import create_dataloader
from utils.hparams import HParam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./data/img_align_celeba/img_align_celeba',
                        help="Data directory for train.")
    parser.add_argument('-i', '--config', type=str, default='',
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None, #-p
                        help="path of checkpoint pt file")
    parser.add_argument('-s', '--save_dir', type=str, default='./checkpoints', help="Save dir for trained model")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    hp = HParam(args.config)
    with open(args.config, 'r', encoding='utf-8') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    # trainloader, testloader = create_dataloader(hp, args.data_dir, face_detector)
    torch.set_num_threads(16)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # from insightface.app import FaceAnalysis
    # from insightface.data import get_image as ins_get_image

    # face_detector = FaceAnalysis(name='buffalo_l')
    # face_detector.prepare(ctx_id=0, det_size=(hp.data.image_size, hp.data.image_size))

    trainloader, testloader = create_dataloader(hp, args.data_dir)
    train(hp, trainloader, testloader, chkpt_path, args.save_dir)
