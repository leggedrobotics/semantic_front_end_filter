from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    n_bins: int =  256
    load_pretrained: bool = True
    input_height: int= 352
    input_width: int =  704
    norm: str = "linear" # 'linear', 'softmax', 'sigmoid'
    min_depth: float = 0.001
    max_depth: float =  40
    min_depth_eval: float = 1e-3
    max_depth_eval: float = 40
    # normalize_output Assume the output of the network is a normalized one, 
    # Use the following param to scale it back
    # i.e. re-mornalized by -m/s, 1/s
    # One way to configure this values is to keep it same with the normalize in class `ToTensor`
    normalize_output_mean: float = 0.120
    normalize_output_std: float = 1.17
    use_adabins: bool = True

@dataclass
class TrainConfig:
    """ 
        ('--epochs', default=25, type=int, help='number of total epochs to run')
        ('--n-bins', '--n_bins', default=80, type=int,
                            help='number of bins/buckets to divide depth range into')
        ('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
        ('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
        ('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
        ('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
        ('--final-div-factor', '--final_div_factor', default=100, type=float,
                            help="final div factor for lr")

        ('--bs', default=16, type=int, help='batch size')
        ('--validate-every', '--validate_every', default=100, type=int, help='validation period')
        ('--gpu', default=None, type=int, help='Which gpu to use')
        ("--name", default="UnetAdaptiveBins")
        ("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                            choices=['linear', 'softmax', 'sigmoid'])
        ("--same-lr", '--same_lr', default=False, action="store_true",
                            help="Use same LR for all param groups")
        ("--distributed", default=False, action="store_true", help="Use DDP if set")
        ("--root", default=".", type=str,
                            help="Root folder to save data in")
        ("--resume", default='', type=str, help="Resume from checkpoint")
        ("--load_pretrained", action="store_true", default=False, help="Load pretrained weights of kitti dataset")

        ("--notes", default='', type=str, help="Wandb notes")
        ("--tags", default='sweep', type=str, help="Wandb tags")

        ("--dataset", default='nyu', type=str, help="Dataset to train on")

        ("--data_path", default='../dataset/nyu/sync/', type=str,
                            help="path to dataset")
        ("--gt_path", default='../dataset/nyu/sync/', type=str,
                            help="path to dataset")

        ('--filenames_file',
                            default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                            type=str, help='path to the filenames text file')

        ('--input_height', type=int, help='input height', default=416)
        ('--input_width', type=int, help='input width', default=544)
        ('--max_depth', type=float, help='maximum depth in estimation', default=10)
        ('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

        ('--do_random_rotate', default=True,
                            help='if set, will perform random rotation for augmentation',
                            action='store_true')
        ('--degree', type=float, help='random rotation maximum degree', default=2.5)
        ('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
        ('--use_right', help='if set, will randomly use right images when train on KITTI',
                            action='store_true')

        ('--data_path_eval',
                            default="../dataset/nyu/official_splits/test/",
                            type=str, help='path to the data for online evaluation')
        ('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                            type=str, help='path to the groundtruth data for online evaluation')
        ('--filenames_file_eval',
                            default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                            type=str, help='path to the filenames text file for online evaluation')

        ('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
        ('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
        ('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                            action='store_true')
        ('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
 """
    bs: int = 8
    dataset: str = "anymal"
    slim_dataset: bool = True # whether or not the dataset is slimed version: (contain projected pc instead of full point cloud information)
    pc_img_input_channel: int = 0 # Which channel of the point cloud image to use, the pc imges have different level of augmentation (slim_dataset is needed)
    pc_img_label_channel: int = 1 # Which channel of the point cloud image to use, the pc imges have different level of augmentation (slim_dataset is needed)

    lr: float =  0.000357
    wd: float =  0.1
    div_factor: int =  25
    final_div_factor: int =  100
    epochs: int =  25
    w_chamfer: float =  0.1
    data_path: str = "extract_trajectories"
    do_random_rotate: bool = True
    degree: float =  1.0

    do_kb_crop: bool = True # if set, crop input images as kitti benchmark images', action='store_true
    garg_crop: bool = True
    eigen_crop: bool=True
    validate_every: int = 100
    same_lr: bool = False
    use_right: bool = False # if set, will randomly use right images when train on KITTI
    pc_image_label_W: float = 0.5
    traj_label_W: float = 1
    traj_distance_variance_ratio = 1/40 # the value used in calculating the variance of traj label. var = (depth*traj_distance_variance_ratio + depth_variance)
    pc_label_uncertainty: bool = False # if yes, use the variance of the label to calculate pc weight
    scale_loss_with_point_number: bool = False # if yes, the loss of each batch is scaled with the number of non-zero values in that batch
    pc_label_mask_traj: bool = True # if yes, the masked for pc_lable loss if masked off where traj label have value
