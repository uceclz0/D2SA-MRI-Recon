# D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction

Official code for our **NeurIPS 2025** paper.

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116601)

Test-time adaptation for accelerated multi-coil MRI reconstruction under an
anatomy distribution shift (knee → brain), built on a U-Net backbone with an
MR-INR (implicit neural representation) prompt.

This repository supports three self-supervised adaptation objectives
(`datafidelity`, `n2n`, `ssdu`), each in two modes:

- `batch`: adapt over a whole patient volume.
- `single`: additionally refine each slice at test time.

## Prerequisites

This project reuses the upstream U-Net code from
[MLI-lab/ttt_for_deep_learning_cs](https://github.com/MLI-lab/ttt_for_deep_learning_cs.git).
Clone it next to this repository (or point `MRI_TTT_ROOT` to its parent folder):

```bash
git clone https://github.com/MLI-lab/ttt_for_deep_learning_cs.git
# optional, if stored elsewhere:
# export MRI_TTT_ROOT=/path/to/parent/of/ttt_for_deep_learning_cs
```

**Datasets and pretrained checkpoints** (`knee_with_self_supervision.pt`,
`brain_with_self_supervision.pt`) are not included here. Please download them by
following the instructions in the upstream repository above.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each script corresponds to one objective/mode combination:

```bash
python scripts/main_functa_datafiedality_knee_to_brain.py \
  --image-dir-base ./outputs \
  --sensitivity-maps-folder /path/to/sens_maps \
  --original-files-folder /path/to/original_files \
  --modality t2 \
  --checkpoint-root /path/to/pretrained_checkpoints
```

Other entry points:

- `main_functa_datafiedality_single_knee_to_brain.py`
- `main_functa_n2n_knee_to_brain.py`
- `main_functa_n2n_single_knee_to_brain.py`
- `main_functa_ssdu_knee_to_brain.py`
- `main_functa_ssdu_single_knee_to_brain.py`

Run any script with `--help` to see all options. Results (reconstructions,
checkpoints, and metric CSVs) are written under `--image-dir-base`.

## Roadmap

- [x] U-Net, anatomy shift (knee → brain)
- [ ] U-Net, other distribution shifts (dataset / modality / acceleration / mask) — coming soon
- [ ] VarNet experiments — coming soon

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2026d2sa,
  title={D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction},
  author={Zhang, Lipei and Sun, Rui and Deng, Zhongying and Cheng, Yanqi and Sch{\"o}nlieb, Carola-Bibiane and Aviles-Rivero, Angelica},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={113189--113221},
  year={2026}
}
```

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgement

This project builds on
[MLI-lab/ttt_for_deep_learning_cs](https://github.com/MLI-lab/ttt_for_deep_learning_cs.git)
and reuses its U-Net components. Please also cite the original work:

> M. Z. Darestani, J. Liu, and R. Heckel,
> *Test-Time Training Can Close the Natural Distribution Shift Performance Gap
> in Deep Learning Based Compressed Sensing.*
