Compile cython modules
```
python setup.py build_ext --inplace
```

# Demo
```
export PYTHONPATH="./":$PYTHONPATH

python demo.py \
--data_path DATA_PATH \
--output_path "./viser_results" \
--cache_path "./cache" \
--sample_interval 1 \
--window_size 30 \
--overlap 10 \
--depth_refine
```

# Evaluation
Change dataset paths within `eval/eval_meta.py` accordingly

## Video Depth

Sintel
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="outputs/sintel_depth" \
--full_seq \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=sintel \
--result_dir="outputs/sintel_depth" \
--output_dir="outputs"
```

Bonn
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=bonn \
--output_dir="outputs/bonn_depth" \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=bonn \
--result_dir="outputs/bonn_depth" \
--output_dir="outputs"
```

KITTI
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=kitti \
--output_dir="outputs/kitti_depth" \
--no_crop \
--flow_loss_weight 0 \
--translation_weight 1e-3

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=kitti \
--result_dir="outputs/kitti_depth" \
--output_dir="outputs"
```

## Camera Pose

Sintel
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="outputs/sintel_pose"
```

ScanNet
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=scannet \
--output_dir="outputs/scannet_pose"
```

TUM
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=tum \
--output_dir="outputs/tum_pose"
```

KITTI Odometry
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3_lc \
--eval_dataset=kitti_odometry \
--output_dir="streaming_pi3_lc_results/kitti_odometry_pose"
```

## MV Recon
Put all datasets in `data/`
```
python mv_recon/eval.py
```
