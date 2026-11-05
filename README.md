# ProbSSL

This repository contains the official implementation of

> Wei & Yuan, *Mitigating Spurious Invariance in Contrastive Learning: A Probabilistic Self-Supervised Learning Framework for Medical Image Analysis*, IEEE Transactions on Medical Imaging, 2026.

- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11494075) · [arXiv](https://arxiv.org/abs/2611.05734)
- Code: [https://github.com/wwzjer/ProbSSL](https://github.com/wwzjer/ProbSSL)

## Installation

```bash
pip install -r requirements.txt
```

## Pre-training

Pre-training writes checkpoints to `pretrain/runs/probssl/`. To run pre-training:

```bash
export DATA_PATH=/path/to/pretrain_data/
bash pretrain/pretrain.sh
```

## Downstream evaluation

Each task under `downstream/` provides `finetune.sh` and `eval.sh`. Set data roots via `DATA_DIR`:

```bash
export DATA_DIR=/path/to/dataset
bash downstream/BTCV/finetune.sh
bash downstream/BTCV/eval.sh

bash downstream/MM-WHS/finetune.sh
bash downstream/MM-WHS/eval.sh

bash downstream/CC-CCII/finetune.sh
bash downstream/CC-CCII/eval.sh

bash downstream/BraTS/finetune.sh
bash downstream/BraTS/eval.sh
```

## Repository structure

```text
ProbSSL/
  README.md
  LICENSE
  requirements.txt
  pretrain/          # self-supervised pre-training
  downstream/
    BTCV/            # multi-organ CT segmentation
    MM-WHS/          # cardiac CT segmentation
    CC-CCII/         # COVID-19 CT classification
    BraTS/           # brain tumor segmentation (MSD Task01)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{wei2026mitigating,
  title={Mitigating Spurious Invariance in Contrastive Learning: A Probabilistic Self-Supervised Learning Framework for Medical Image Analysis},
  author={Wei, Wei and Yuan, Yading},
  journal={IEEE Transactions on Medical Imaging},
  volume={45},
  number={7},
  pages={3862--3877},
  year={2026},
  doi={10.1109/TMI.2026.3687158}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
