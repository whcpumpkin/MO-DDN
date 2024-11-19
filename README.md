# MO-DDN: A Coarse-to-Fine Attribute-based Exploration Agent for Multi-Object Demand-driven Navigation
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/moddn)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2410.03488)

This repo is the official implementation of NeurIPS 2024 paper, [Multi-Object Demand-driven Navigation](https://arxiv.org/abs/2410.03488)

## TODOs (Under Development):
- [ ] README
- [x] Environment Code
- [ ] Modified HSSD Dataset
- [x] Instruction Dataset
- [ ] Trajectory Dataset
- [ ] Utils Code
- [ ] Training
- [ ] Testing

## Warning: I am currently refactoring my code. Although all code committed to the repository has been tested, there may still be some minor issues.  Feel free to ask any questions.

## Download
We modified the HSSD dataset to include more object labels. You can download the modified dataset from [OneDrive]() or [百度网盘]().



## Installation
We use conda to manage our environment.

```
conda create -n moddn python=3.9
conda activate moddn
```

We use habitat-lab and habitat-sim as our simulation environment.
```
cd habitat-lab/habitat-lab
pip install -e .
conda install habitat-sim=0.2.5 withbullet headless -c conda-forge -c aihabitat
```



## Contact
If you have any suggestion or questions, please feel free to contact us:

[Hongcheng Wang](https://whcpumpkin.github.io): [whc.1999@pku.edu.cn](mailto:whc.1999@pku.edu.cn)

Peiqi Liu: [peiqiliu@stu.pku.edu.cn](mailto:peiqiliu@stu.pku.edu.cn)

[Hao Dong](https://zsdonghao.github.io/): [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn)
****
## Citation

```bibtex
@inproceedings{wang2024mo,
  title={MO-DDN: A Coarse-to-Fine Attribute-based Exploration Agent for Multi-object Demand-driven Navigation},
  author={Wang, Hongcheng and Liu, Peiqi and Cai, Wenzhe and Wu, Mingdong and Qian, Zhengyu and Dong, Hao},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Thanks
the fold of ''habitat-lab''  is modified from https://github.com/facebookresearch/habitat-lab, with version 0.2.5.

we extend the object label metadata in the HSSD dataset and use it in our project. Please see [issue](https://github.com/3dlg-hcvc/hssd/issues/13) for the modification details. The original HSSD dataset can be found at https://github.com/3dlg-hcvc/hssd. 