Compile cython modules
```
python setup.py build_ext --inplace
```

## Video Depth

Sintel
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="streaming_pi3_results/sintel_depth" \
--full_seq \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=sintel \
--result_dir="streaming_pi3_results/sintel_depth" \
--output_dir="streaming_pi3_results"
```

Bonn
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=bonn \
--output_dir="streaming_pi3_results/bonn_depth" \
--no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=bonn \
--result_dir="streaming_pi3_results/bonn_depth" \
--output_dir="streaming_pi3_results"
```

KITTI
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=kitti \
--output_dir="streaming_pi3_results/kitti_depth" \
--no_crop \
--flow_loss_weight 0 \
--translation_weight 1e-3

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
--eval_dataset=kitti \
--result_dir="streaming_pi3_results/kitti_depth" \
--output_dir="streaming_pi3_results"
```

## Camera Pose

Sintel
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=sintel \
--output_dir="streaming_pi3_results/sintel_pose"
```

ScanNet
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=scannet \
--output_dir="streaming_pi3_results/scannet_pose"
```

TUM
```
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3 \
--eval_dataset=tum \
--output_dir="streaming_pi3_results/tum_pose"
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
