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
- [x] Utils Code
- [x] Training
- [x] Testing

##  I'm very sorry that due to personal reasons, I was unable to release all the weight and evaluation codes in a timely manner. If there are any issues with the current code, please feel free to contact me.

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

For pretrained model, we provide the following links:

百度网盘: https://pan.baidu.com/s/1eLXxMUqsImlf47Sw71dVjw?pwd=hgg9

Google Drive: https://drive.google.com/drive/folders/1nKVcWHqgwyRCp5yZWY6-Wsf6ZP2E7o2x?usp=sharing

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
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
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
python data_collection.py --running_mode=local_data_collection --workers=N --epoch=M --save_dir=path/to/save/dir --save_name=MO-DDN_local
```

This means that you will launch N environments, each collecting M local trajectories, for a total of N*M. Please replace `N` and `M` with two suitable integers depending on your machine's memory and CPU. For reference, my machine consists of two E5 2680V4, 128G RAM, one RTX 2080Ti 22G graphics card, and I set N=16, M=3100, which consumes about 24 hours or so to collect ~50000 local trajectories. Please note that this data can consume around 3T of space, make sure you have the space to store it.

To collect `full trajectory`, run 
```
python data_collection.py --running_mode=full_data_collection --workers=N --epoch=M --save_dir=path/to/save/dir --save_name=VTN_full
```

For reference, my machine consists of two E5 2680V4, 128G RAM, one RTX 2080Ti 22G graphics card, and I set N=12, M=200, which consumes about 12 hours or so to collect ~20000 full trajectories.


## Training

To train the attribute model, run
```
python train_attribute_model.py --save_dir=path/to/save/dir --save_name=attribute_model --recon_coef=1.0 --vq_coef=1.0 --commit_coef=0.25 --matching_coef=1 --obj_attribute_coef=2 --ins_attribute_coef=2
```

then obtain a model named `attribute_model_best.pth` in the `$save_dir$/$save_name$` directory.

To train the Fine-Exploration Module, first generate the trajectory matadata by running
```
python data_collection.py --running_mode=generate_metadata --path_to_raw_traj=path/to/save/dir/MO-DDN_local/Year-Month-Day_Hour-Minute-Second
```

Then put the `attribute_model_best.pth` into `pretrained_models` and run
```
python train_fine_model.py --save_dir=path/to/save/dir --save_name=fine_exploration_module --epoch=30
```
then obtain fine_model_X.pth in the `$save_dir$/$save_name$/y-m-d-h-m` directory, where X is the epoch number. See the `eval_results.txt` file in the same directory for the evaluation results and pick the highest validation accuracy model for the next step.


## Evaluation

Before evaluation, you should download `GLEE_SwinL_Scaleup10m.pth` and put GLEE model into `GLEE_model` file.
Download `openai/clip-vit-base-patch32` model, and put this model into `thirdparty/GLEE/`



- **Script**: `evaluation_full_pipeline.py` — runs the full coarse-to-fine evaluation loop and saves metrics.

- **Example** (run from repository root):

```bash
python evaluation_full_pipeline.py \
  --eval_model_path pre_trained_models/il_agent.pth \
  --task_mode val \
  --epoch 500 \
  --max_step 300 \
  --device cuda
```

- **Where results go**: a JSON summary is written to `<save_dir>/<timestamp>/eval_metrics.json` (default `--save_dir` is `LLM+Fine`).

- **Common arguments (short)**:
  - `--eval_model_path`: path to a `.pth` checkpoint (file or folder). If a folder is provided the script will try common names or the latest `.pth` inside.
  - `--task_mode`: `train` or `val` (which dataset split to run).
  - `--epoch`: number of episodes to run (controls how many tasks are evaluated).
  - `--max_step`: max steps per episode before forcing termination.
  - `--device`: `cuda` or `cpu` (device used for models/compute).
  - `--save_dir`: base directory to write per-run outputs and metrics.
  - `--random_fine`: set to `1` to skip loading a trained fine-exploration agent and use random fine behavior.
  - `--add_noise`: set to `1` to add small RGB/depth noise during evaluation (debugging/robustness).
  - `--seed`: random seed for reproducible runs.

- **Hints**:
  - Make sure your conda env is activated and `habitat-lab` / `habitat-sim` are installed as described in Installation.
  - To reduce simulator logs set `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet` in your shell.
  - See `utils/args.py` for a full list of available options and defaults.



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