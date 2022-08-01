from .cfg import TrainConfig, ModelConfig
from dataclasses import asdict

def parse_args(parser, flatten = False):

    parser.add_arguments(TrainConfig, dest="trainconfig")
    parser.add_arguments(ModelConfig, dest="modelconfig")
    
    args = parser.parse_args()
    args.batch_size = args.trainconfig.bs
    args.num_threads = args.trainconfig.workers
    args.mode = 'train'
    args.data_path = args.trainconfig.data_path
    args.min_depth = args.modelconfig.min_depth
    args.max_depth = args.modelconfig.max_depth
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