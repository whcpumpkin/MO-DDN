# MO-DDN: A Coarse-to-Fine Attribute-based Exploration Agent for Multi-Object Demand-driven Navigation
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/moddn)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2410.03488)

This repo is the official implementation of NeurIPS 2024 paper, [Multi-Object Demand-driven Navigation](https://arxiv.org/abs/2410.03488)

## TODOs (Under Development):
- [ ] README
- [x] Environment Code
- [x] Modified HSSD Dataset
- [x] Instruction Dataset
- [x] Benchmark Demo
- [x] Trajectory Collecting Code
- [ ] Utils Code
- [ ] Training
- [ ] Testing

## Warning: I am currently refactoring my code. Although all code committed to the repository has been tested, there may still be some minor issues.  Feel free to ask any questions.

## Download
We modified the HSSD dataset to include more object labels. You can download the modified HSSD dataset from [Hugging Face](https://huggingface.co/datasets/whcpumpkin/moddn-hssd) or [百度网盘](https://pan.baidu.com/s/1eLXxMUqsImlf47Sw71dVjw?pwd=hgg9).
You will download a zip file or three splited zip files containing the dataset.

```
# Download from Hugging Face
cd MO-DDN/data
git clone https://github.com/whcpumpkin/moddn-hssd.git
mv ./moddn-hssd/scene_dataset* ./
cat scene_datasets.zip.001 scene_datasets.zip.002 scene_datasets.zip.003 > scene_datasets_combined.zip
unzip scene_datasets_combined.zip

# Download from 百度网盘
cd MO-DDN/data
cp path/to/your/downloaded/scene_dataset.zip ./
unzip scene_dataset.zip
```

## Dataset Structure
The dataset, including the modified HSSD dataset and the instruction dataset, is organized as follows(in the `MO-DDN/data` directory):
```
┌data
│ ├datasets
│ │  └ddnplus
│ │      └hssd-hab_v0.2.5
│ │         ├scene_object_navigable_point
│ │         ├train
│ │         └val
│ └scene_datasets
│    └hssd-hab
│        ├objects
│        ├scene_filter_files
│        ├scenes
│        └...

```



## Installation
We use conda to manage our environment.

```
conda create -n moddn python=3.9 cmake=3.14
conda activate moddn
```

We use habitat-lab and habitat-sim as our simulation environment.
```
cd habitat-lab/habitat-lab
pip install -e .
conda install habitat-sim=0.2.5 withbullet headless -c conda-forge -c aihabitat
```

Then, install other dependencies:
```
pip install -r requirements.txt
```

## Demo
We provide a very simple demo to show how to use our benchmark.
```
python demo.py --task_mode=train --scene_mode=train
```
`--task_mode` can be set to `train` or `val`. `--scene_mode` can be set to `train` or `val`. `train` mode means `seen` in our paper, while `val` mode means `unseen` in our paper.

To ignore the habitat-sim log, you can set the following environment variables:

```
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
```

## Trajectory Collecting Code

In MO-DDN, we collect two different kind of trajectories, `local trajectory` and `full trajectory`. We use `local trajectory` to train our Fine-Exploration Module and use `full trajectory` to train the VTN baseline. For more details, please see Sec A.3.2 and A.4.2 in our supplemental material. 


To collect `local trajectory`, run 
```
python data_collection.py --running_mode=local_data_collection --workers=N --epoch=M --save_dir=path/to/save/dir --save_name=MODDN_local
```

This means that you will launch N environments, each collecting M local trajectories, for a total of N*M. Please replace `N` and `M` with two suitable integers depending on your machine's memory and CPU. For reference, my machine consists of two E5 2680V4, 128G RAM, one RTX 2080Ti 22G graphics card, and I set N=16, M=3100, which consumes about 24 hours or so to collect ~50000 local trajectories. Please note that this data can consume around 3T of space, make sure you have the space to store it.

To collect `full trajectory`, run 
```
python data_collection.py --running_mode=full_data_collection --workers=N --epoch=M --save_dir=path/to/save/dir --save_name=VTN_full
```

For reference, my machine consists of two E5 2680V4, 128G RAM, one RTX 2080Ti 22G graphics card, and I set N=12, M=200, which consumes about 12 hours or so to collect ~20000 full trajectories.

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