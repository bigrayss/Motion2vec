## Environment
The project is developed under the following environment:
- Python 3.8.10
- PyTorch 2.0.0
- CUDA 12.2

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 
## Dataset
### Human3.6M
#### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For MotionAGFormer-Base and MotionAGFormer-Large**:
```text
python h36m.py  --n-frames 243
```

**For MotionAGFormer-Small**:
```text
python h36m.py --n-frames 81
```

**For MotionAGFormer-XSmall**:
```text
python[data](data) h36m.py --n-frames 27
```

#### Visualization
Run the following command in the `data/preprocess` directory (it expects 243 frames):
```text
python visualize.py --dataset h36m --sequence-number <AN ARBITRARY NUMBER>
```
This should create a gif file named `h36m_pose<SEQ_NUMBER>.gif` within `data` directory.

### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.
#### Visualization
Run it same as the visualization for Human3.6M, but `--dataset` should be set to `mpi`.
## Pretraining
After dataset preparation, you can pretrain the model as follows:
### Human3.6M
You can pretrain Human3.6M with the following command:
```
CUDA_VISIBLE_DEVICES=1 python pre_train.py --new-checkpoint checkpoint/pretrain --config configs/pretrain/Pose2Vec.yaml > log/pretrain.log
```
where config files are located at `configs/pretrain`
## Training
After pretraining, you can train the model as follows:
### Human3.6M
You can pretrain Human3.6M with the following command:
```
python train.py --P2V_reload 1 -c checkpoint_pretrain/pretrain --checkpoint-file best_epoch.pth.tr --new-checkpoint checkpoint/finetune --config configs/h36m/MotionAGFormer-new.yaml > log/finetune.log
```
where config files are located at `configs/h36m`.
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`. Like Human3.6M, weight and biases can be used.
## Evaluation
| Method            | # frames  | # Params | # MACs         | H3.6M weights | MPI-INF-3DHP weights |
|-------------------|-----------|----------|----------------|---------------|----------------------|
| MotionAGFormer-XS |     27    |   2.2M  |      1.0G     |    [download](https://drive.google.com/file/d/1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH/view?usp=sharing)   |       [download](https://drive.google.com/file/d/1FebsaA-DGqB0Pba_dVXCQDedfG13Zu5O/view?usp=sharing)      |
| MotionAGFormer-S  |     81    |   4.8M  |      6.6G     |    [download](https://drive.google.com/file/d/1DrF7WZdDvRPsH12gQm5DPXbviZ4waYFf/view?usp=sharing)   |       [download](https://drive.google.com/file/d/1RzTGl_9d_wm34ZnBqkQWqy22kOOq0Dx5/view?usp=sharing)       |
| MotionAGFormer-B  | 243 \| 81 |  11.7M  | 48.3G \| 16G |    [download](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view?usp=drive_link)   |       [download](https://drive.google.com/file/d/18Pz0qQp7DvrrvDNqFtem5O5cBHBBAGg-/view?usp=sharing)      |
| MotionAGFormer-L  | 243 \| 81 |  19.0M  | 78.3G \| 26G |    [download](https://drive.google.com/file/d/1WI8QSsD84wlXIdK1dLp6hPZq4FPozmVZ/view?usp=sharing)   |       [download](https://drive.google.com/file/d/10am2CelOV5Nt2NDhcEEMgFpdcuKN3J3G/view?usp=sharing)       |

After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
For example if MotionAGFormer-L of H.36M is downloaded and put in `checkpoint` directory, then we can run:
```
python train.py --eval-only --checkpoint checkpoint --checkpoint-file motionagformer-l-h36m.pth.tr --config configs/h36m/MotionAGFormer-large.yaml
```
Similarly, MPI-INF-3DHP can be evaluated as follows:
```
python train_3dhp.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
## Demo
Our demo is a modified version of the one provided by [MHFormer](https://github.com/Vegetebird/MHFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.

Run the command below:
```
python demo/vis.py --video sample_video.mp4
```
## Acknowledgement
Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)

We thank the authors for releasing their codes.
## Citation
If you find our work useful for your project, please consider citing the paper: