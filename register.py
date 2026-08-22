import world
from dataloader import Loader
from model_v1 import LightGCN, PureMF, MultiModalLightGCN
from utils import build_item_knn_graph, load_or_build_item_graph

def dataset_factory():
    return Loader(world)

MODELS = {
    'mf': PureMF,
    'lgn': LightGCN,
    'lgn_mm': MultiModalLightGCN,  # wrapper, constructed specially below
}


def model_factory(dataset):
    mname = world.config.model
    dev = world.config.dev

    if mname == 'lgn':
        # return LightGCN(world.config, dataset)
        model = LightGCN(world.config, dataset)
        return model.to(dev)
    if mname == 'mf':
        # return PureMF(world.config, dataset)
        model = PureMF(world.config, dataset)
        return model.to(dev)
    if mname == 'lgn_mm':
        # construct CF backbone first
        cf = LightGCN(world.config, dataset).to(dev)
        dev = world.config.dev
        # prepare item graphs
        g_txt = load_or_build_item_graph(
            feat_path=world.config.text_feat,
            graph_path=world.config.item_text_graph,
            k=world.config.item_knn_k,
            device=dev,
            force_build=world.config.force_build_item_knn
        )
        print('g_txt: ', g_txt)
        g_img = load_or_build_item_graph(
            feat_path=world.config.image_feat,
            graph_path=world.config.item_image_graph,
            k=world.config.item_knn_k,
            device=dev,
            force_build=world.config.force_build_item_knn
        )
        print('g_img: ', g_img)
        if world.config.audio_feat:
            g_aud = load_or_build_item_graph(
                feat_path=world.config.audio_feat,
                graph_path=world.config.item_audio_graph,
                k=world.config.item_knn_k,
                device=dev,
                force_build=world.config.force_build_item_knn
                )
            print('g_aud: ', g_aud)
            mm = MultiModalLightGCN(world.config, dataset, cf_backbone=cf, g_text=g_txt, g_image=g_img, g_audio=g_aud)

        elif world.config.user_text_feat:
            g_user = load_or_build_item_graph(
                feat_path=world.config.user_text_feat,
                graph_path=world.config.user_text_graph,
                k=world.config.item_knn_k,
                device=dev,
                force_build=world.config.force_build_item_knn
                )
            print('g_user: ', g_user)
            mm = MultiModalLightGCN(world.config, dataset, cf_backbone=cf, g_text=g_txt, g_image=g_img, g_user=g_user)

        else:
            mm = MultiModalLightGCN(world.config, dataset, cf_backbone=cf, g_text=g_txt, g_image=g_img)

        return mm.to(dev)
    raise ValueError(f"Unknown model: {mname}")
