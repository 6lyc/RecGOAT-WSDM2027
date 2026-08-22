# RecGOAT
The Official implementation of our paper "RecGOAT: Graph Optimal Adaptive Transport for LLM-Enhanced Multimodal Recommendation with Dual Semantic Alignment"

[The-Supplementary-Material-for-RecGOAT]()

## Abstract
Large models (LMs), including language, vision, and multimodal models, provide strong semantic understanding, contextual reasoning, and cross-modal fusion for multimodal recommendation. However, LM representations are inherently optimized for general semantic tasks, while recommendation models rely heavily on sparse user/item unique identity (ID) features. Existing works overlook the fundamental representational divergence between large models and recommendation systems, resulting in incompatible multimodal representations and suboptimal recommendation performance. To bridge this gap, we propose \textbf{RecGOAT}, a novel yet simple dual semantic alignment framework for LLM-enhanced multimodal recommendation, which offers theoretically guaranteed alignment capability. RecGOAT first employs graph attention networks to enrich collaborative signals by modeling item-item, user-item, and user-user relationships, leveraging user/item LM representations and interaction history. Furthermore, we design a dual-granularity progressive multimodality-ID alignment framework, which achieves instance-level and distribution-level semantic alignment via cross-modal contrastive learning (CMCL) and optimal adaptive transport (OAT), respectively. Theoretically, we demonstrate that the unified representations derived from our alignment framework exhibit superior semantic consistency and comprehensiveness. Extensive experiments on three public benchmarks show that our RecGOAT achieves state-of-the-art performance, empirically validating our theoretical insights.


<div align="center">
<img width="1088" height="666" alt="RecGOAT" src="https://github.com/6lyc/RecGOAT-WSDM2027/blob/main/framework.png" />
</div>

## Dependence

To install the dependencies: 

```sh
pip install -r requirements.txt
```

## Usage and Hyperparameter

```bash
# Baby
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset baby_raw --model lgn_mm --use_multimodal 1 --text_feat ./data/baby_raw/text_feat.npy --image_feat ./data/baby_raw/image_feat.npy --item_knn_k 20 --item_branch_layers 1 --epochs 1000 --recdim 800 --layer 3 --lr 8e-4 --bpr_batch 4096 --topks "[10]" --fusion oat --contrastive_weight 0.032 > ./baby.log 2>&1 &

# Sports
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset sports_raw --model lgn_mm --use_multimodal 1 --text_feat ./data/sports_raw/text_feat.npy --image_feat ./data/sports_raw/image_feat.npy --item_knn_k 30 --item_branch_layers 1 --epochs 1000 --recdim 600 --layer 3 --lr 8e-4 --bpr_batch 4096 --topks "[10]" --fusion oat --contrastive_weight 0.005 > ./sports.log 2>&1 &

# Electronics
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset electronics --model lgn_mm --use_multimodal 1 --text_feat ./data/electronics_raw/text_feat.npy --image_feat ./data/electronics_raw/image_feat.npy --item_knn_k 20 --item_branch_layers 1 --epochs 2000 --recdim 200 --layer 3 --lr 8e-4 --bpr_batch 4096 --topks "[10]" --fusion oat --contrastive_weight 0.05 > ./electronics.log 2>&1 &
```
