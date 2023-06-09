from dataclasses import dataclass, field, asdict
import os

@dataclass
class ModelConfig:
    input_channel: int = 4
    load_pretrained: bool = False
    input_height: int= 500
    input_width: int =  720
    norm: str = "linear" # 'linear', 'softmax', 'sigmoid'
    min_depth: float = 0.001
    max_depth: float =  10
    min_depth_eval: float = 1e-3
    max_depth_eval: float = 10
    max_pc_depth: float = 15
    # normalize_output Assume the output of the network is a normalized one, 
    # Use the following param to scale it back
    # i.e. re-mornalized by -m/s, 1/s
    # One way to configure this values is to keep it same with the normalize in class `ToTensor`
    normalize_output_mean: float = 0.120
    normalize_output_std: float = 1.17
    deactivate_bn: bool = True
    skip_connection: bool = False
    interpolate_mode : str = "bilinear" # "bilinear" or "convT", define what is used in upsampling of decoder 
    output_mask : bool = True # Please keep this true
    # output_mask_channels : int = 1
    decoder_num: int = 1 # One or two
    ablation: str = "" # "onlyRGB", "onlyPC"

@dataclass
class TrainConfig:
    wandb_name: str = "random"
    bs: int = 6
    workers: int = 16 # Number of workers for data loading
    dataset: str = "anymal"
    slim_dataset: bool = True # whether or not the dataset is slimed version: (contain projected pc instead of full point cloud information)
    pc_img_input_channel: int = 0 # Which channel of the point cloud image to use, the pc imges have different level of augmentation (slim_dataset is needed)
    pc_img_label_channel: int = 0 # Which channel of the point cloud image to use, the pc imges have different level of augmentation (slim_dataset is needed)
    lr: float =  0.000357
    wd: float =  0.1
    div_factor: int =  25 
    final_div_factor: int =  100
    epochs: int =  30
    w_chamfer: float =  0.1
    data_path: str = None
    camera_cali_path: str = "/semantic_front_end_filter/Labelling/Example_Files/alphasense"
    do_random_rotate: bool = True
    degree: float =  1.0

    do_kb_crop: bool = True # if set, crop input images as kitti benchmark images', action='store_true
    garg_crop: bool = True
    eigen_crop: bool=True
    random_crop: bool=True
    random_flip: bool=True
    traj_variance_threashold: float = 1 # trajectory label will be filtered by this thershold # if the variance is below this above this value, mask the corresponding traj label off
    validate_every: int = 100
    same_lr: bool = True
    use_right: bool = False # if set, will randomly use right images when train on KITTI
    pc_image_label_W: float = 1 # pc_image_label_W and traj_label_W_4mask are used to control the crossentropy loss for the mask
    traj_label_W_4mask: float = 1
    traj_label_W: float = 0.02 # 0.002
    mask_loss_W: float = 1
    mask_weight_mode: str='sigmoid' # binary or sigmoid
    filter_image_before_loss: bool = True
    sprase_traj_mask: bool = False # True, if you want to train with support surface mask filtered by the pc label, 
                                   # only in this brach it will set the model to predict the delta depth
    mask_ratio: float = 1 # Expected ratio of mask_ground/mask_nonground

    traj_distance_variance_ratio: float = 0 # the value used in calculating the variance of traj label. var = (depth*traj_distance_variance_ratio + depth_variance)
    pc_label_uncertainty: bool = False # if yes, use the variance of the label to calculate pc weight
    scale_loss_with_point_number: bool = True # if yes, the loss of each batch is scaled with the number of non-zero values in that batch
    
    train_with_sample: bool = False # if yes, the training set will be same as the testing set, contains only two trajectories
    testing: list = field(default_factory=lambda: [ "Reconstruct_2022-07-19-18-16-39_0", # Perugia high grass
                                                    "Reconstruct_2022-07-19-19-02-15_0", # Perugia forest
                                                    "Reconstruct_2022-07-18-20-34-01_0", # Perugia grassland
                                                    # "Reconstruct_2022-04-25-15-31-34_0", # South Africa
                                                    # "Reconstruct_2022-04-26-16-34-01_0", # South Africa 
                                                    # "Reconstruct_2022-04-26-17-35-27_0", # South Africa 
                                                    # "Reconstruct_2022-04-26-17-05-24_0"  # South Africa 
                                                    ]) 
    training: list = field(default_factory=lambda: ["Reconstruct_2022-07-19-20-46-08_0", # Perugia high grass
                                                    "Reconstruct_2022-07-21-10-47-29_0", # Perugia forest
                                                    "Reconstruct_2022-07-19-20-06-22_0", # Perugia grassland
                                                    # "Reconstruct_2022-04-25-15-31-34_0", # South Africa
                                                    # "Reconstruct_2022-04-26-16-34-01_0", # South Africa 
                                                    # "Reconstruct_2022-04-26-17-35-27_0", # South Africa 
                                                    # "Reconstruct_2022-04-26-17-05-24_0"  # South Africa 
                                                    ]) 



def parse_args(parser, flatten = False, argstr = None):

    parser.add_arguments(TrainConfig, dest="trainconfig")
    parser.add_arguments(ModelConfig, dest="modelconfig")
    
    args, _ = parser.parse_known_args() if argstr is None else parser.parse_args(argstr)
    args.batch_size = args.trainconfig.bs
    args.num_threads = args.trainconfig.workers
    args.mode = 'train'
    args.data_path = args.trainconfig.data_path
    args.min_depth = args.modelconfig.min_depth
    args.max_depth = args.modelconfig.max_depth
    args.max_pc_depth = args.modelconfig.max_pc_depth
    args.min_depth_eval = args.modelconfig.min_depth_eval
    args.max_depth_eval = args.modelconfig.max_depth_eval
    args.load_pretrained = args.modelconfig.load_pretrained

    args.chamfer = args.trainconfig.w_chamfer > 0

    if(flatten):
        # flatten nested configs, to make it easier to wandb
        for k,v in asdict(args.trainconfig).items():
            setattr(args, f"trainconfig:{k}", v)
        for k,v in asdict(args.modelconfig).items():
            setattr(args, f"modelconfig:{k}", v)

    return args