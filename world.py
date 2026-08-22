import os
import torch

class Config:
    def __init__(self, args):
        self.dataset = args.dataset
        self.model = args.model.lower()
        self.topks = eval(args.topks) if isinstance(args.topks, str) else args.topks
        self.TEST_U_BATCH_SIZE = args.test_u_batch_size
        self.multicore = args.multicore
        self.recdim = args.recdim
        self.layer = args.layer
        self.lr = args.lr
        self.decay = args.decay
        self.epochs = args.epochs
        self.bpr_batch = args.bpr_batch
        self.dev = torch.device('cuda' if torch.cuda.is_available() and args.cuda and args.gpu_id >= 0 else 'cpu')
        self.seed = args.seed
        self.tensorboard = args.tensorboard
        self.A_split = args.a_split
        self.A_n_fold = args.a_fold
        self.path = args.path
        self.pretrain = args.pretrain

        # multimodal
        self.use_multimodal = bool(args.use_multimodal)
        self.text_feat = args.text_feat
        self.user_text_feat = args.user_text_feat
        self.image_feat = args.image_feat
        self.audio_feat = args.audio_feat
        self.item_text_graph = args.item_text_graph
        self.user_text_graph = args.user_text_graph
        self.item_image_graph = args.item_image_graph
        self.item_audio_graph = args.item_audio_graph
        self.force_build_item_knn = bool(args.force_build_item_knn)
        self.item_knn_k = args.item_knn_k
        self.item_branch_layers = args.item_branch_layers
        self.user_branch_layers = args.user_branch_layers
        self.fusion = args.fusion
        self.attn_heads = args.attn_heads
        self.attn_dropout = args.attn_dropout
        self.fuse_drop = args.fuse_drop

        # OT
        self.alpha = args.alpha
        self.beta = args.beta
        self.eta = args.eta

        # CL
        self.contrastive_weight = args.contrastive_weight
        self.temperature = args.temperature
        self.modality_consistency_weight = args.modality_consistency_weight

    def as_dict(self):
        return self.__dict__


def cprint(words: str):
    print(words)


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# dataset root resolver
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def dataset_path(dataset: str):
    # original repo expects ../data/<dataset>/
    print(os.path.abspath(os.path.join(ROOT_PATH, '..', 'RecGOAT_SIGIR2026_Submission/data', dataset)))
    return os.path.abspath(os.path.join(ROOT_PATH, '..', 'RecGOAT_SIGIR2026_Submission/data', dataset))
