<div align="center">
<h1>LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction</h1>
<a href="http://arxiv.org/abs/2512.13680"><img src="https://img.shields.io/badge/arXiv-2512.13680-b31b1b" alt="arXiv"></a>
<a href="https://neu-vi.github.io/LASER/"><img src="https://img.shields.io/badge/Project-Website-orange" alt="Project Page"></a>

[Tianye Ding<sup>1*</sup>](https://jerrygcding.github.io/), 
[Yiming Xie<sup>1*</sup>](https://ymingxie.github.io/), 
[Yiqing Liang<sup>2*</sup>](https://lynl7130.github.io/), 
[Moitreya Chatterjee<sup>3</sup>](https://sites.google.com/site/metrosmiles/), 
[Pedro Miraldo<sup>3</sup>](https://pmiraldo.github.io/), 
[Huaizu Jiang<sup>1</sup>](https://jianghz.me/)\
<sup>1</sup> Northeastern University, <sup>2</sup> Independent Researcher, <sup>3</sup> Mitsubishi Electric Research Laboratories\
<sup>*</sup> Equal Contribution
</div>

## 📢 Updates
* **[2025-12-15]** ArXiv preprint released.

## 📝 To-Do List

- [x] Release framework codebase
- [x] Release inference code
- [x] Add data preparation instruction
- [x] Release evaluation code
- [x] Add Viser integration
- [ ] Release loop-closure demo

## 💡 Abstract
We propose LASER, a training-free framework that converts an offline reconstruction model into a streaming system by aligning predictions across consecutive temporal windows. 
We observe that simple similarity transformation (Sim(3)) alignment fails due to layer depth misalignment: monocular scale ambiguity causes relative depth scales of different scene layers to vary inconsistently between windows. 
To address this, we introduce layer-wise scale alignment, which segments depth predictions into discrete layers, computes per-layer scale factors, and propagates them across both adjacent windows and timestamps.

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone --recursive git@github.com:neu-vi/LASER.git
cd LASER

# 2. Create environment
conda create -n laser -y python=3.11
conda activate laser

# 3. Install dependencies
pip install -r requirements.txt

# 4. Compile cython modules
python setup.py build_ext --inplace

# 5. Install Viser
pip install -e viser
```

(Optional) Download checkpoints needed for loop-closure inference

```bash
bash ./scripts/download_weights.sh
```

## 🚀 Usage

### Inference
To run the inference code, you can use the following command:
```bash
export PYTHONPATH="./":$PYTHONPATH

python demo.py \
--data_path DATA_PATH \
--output_path "./viser_results" \
--cache_path "./cache" \
--sample_interval SAMPLE_INTERVAL \
--window_size WINDOW_SIZE \
--overlap OVERLAP \
--depth_refine

# example inference script
python demo.py \
--data_path "examples/titanic" \
--output_path "./viser_results" \
--cache_path "./cache" \
--sample_interval 1 \
--window_size 30 \
--overlap 10 \
--depth_refine
```
The results will be saved in the `viser_results/SEQ_NAME`directory for future visualization.

### Visualization
To visualize the interactive 4D results, you can use the following command:
```bash
python viser/visualizer_monst3r.py --data viser_results/SEQ_NAME

# example visualization script
python viser/visualizer_monst3r.py --data viser_results/titanic
```

## Evaluation
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/prepare_training.md#dataset-setup) for dataset setup details.

Put all datasets in `data/`.

### Video Depth

Sintel
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="outputs/video_depth/sintel_depth" \
--full_seq \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=sintel \
--result_dir="outputs/video_depth/sintel_depth" \
--output_dir="outputs/video_depth"
```

Bonn
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=bonn \
--output_dir="outputs/video_depth/bonn_depth" \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=bonn \
--result_dir="outputs/video_depth/bonn_depth" \
--output_dir="outputs/video_depth"
```

KITTI
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=kitti \
--output_dir="outputs/video_depth/kitti_depth" \
--no_crop \
--flow_loss_weight 0 \
--translation_weight 1e-3

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=kitti \
--result_dir="outputs/video_depth/kitti_depth" \
--output_dir="outputs/video_depth"
```

### Camera Pose

Sintel
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="outputs/cam_pose/sintel_pose"
```

ScanNet
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=scannet \
--output_dir="outputs/cam_pose/scannet_pose"
```

TUM
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=tum \
--output_dir="outputs/cam_pose/tum_pose"
```
<!-- 
KITTI Odometry
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3_lc \
--eval_dataset=kitti_odometry \
--output_dir="outputs/cam_pose/kitti_odometry_pose"
```

### MV Recon
```bash
export PYTHONPATH="./":$PYTHONPATH
python mv_recon/eval.py
``` -->

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@article{ding2025laser,
  title={LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction},
  author={Ding, Tianye and Xie, Yiming and Liang, Yiqing and Chatterjee, Moitreya and Miraldo, Pedro and Jiang, Huaizu},
  year={2025}
}
```

## Acknowledgements
We would like to thank the authors for the following excellent open source projects:
[VGGT](https://github.com/facebookresearch/vggt/tree/main), 
[&pi;<sup>3</sup>](https://github.com/yyfz/Pi3),
[MonST3R](https://github.com/Junyi42/monst3r),
[CUT3R](https://github.com/CUT3R/CUT3R),
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long/tree/main)
and many other inspiring works in the community.
