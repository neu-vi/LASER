import torch

from pi3.models.pi3 import Pi3
from inference_engine import StreamingWindowEngine
from utils.load_fn import load_and_preprocess_images
from eval.save_func import save_for_viser

import os
import argparse
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def get_args_parser():
    parser = argparse.ArgumentParser('Depth metric evaluation', add_help=False)
    parser.add_argument('--model_ckpt', default=None, type=str, help='checkpoint to load model')
    parser.add_argument('--data_path', type=str, help='sequence data path')
    parser.add_argument('--scene_name', default=None, type=str, help='scene_name')
    parser.add_argument('--cache_path', default='./inference_cache', type=str,
                        help='output inference cache')
    parser.add_argument('--output_path', default='./viser_results', type=str,
                        help='output visualization results')
    parser.add_argument('--sample_interval', default=1, type=int, help='sequence sample interval')
    parser.add_argument('--window_size', default=10, type=int, help='sliding window size')
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')
    parser.add_argument('--depth_refine', action='store_true', help='enable depth refine')

    return parser


def load_model(args):
    # model
    if args.model_ckpt:
        model = Pi3().to(device)
        print('Loading checkpoint: ', args.model_ckpt)
        ckpt = torch.load(args.model_ckpt, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=True))
        del ckpt
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device)

    return StreamingWindowEngine(
        model,
        inference_device=device,
        dtype=dtype,
        window_size=args.window_size,
        overlap=args.overlap,
        cache_root=args.cache_path,
        depth_refine=args.depth_refine,
        top_conf_percentile=0.3
    )


def run_model(image_names, scene_name, output_path):
    images = load_and_preprocess_images(image_names).to(device)
    image_windows = model.img_sliding_window(images)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    model.begin()

    start_ev.record()
    for sample in tqdm(image_windows, 'Window inference'):
        model(sample)
    model.end()
    end_ev.record()

    save_dict = model.parse_inference_cache_summary()
    for key in save_dict.keys():
        if isinstance(save_dict[key], torch.Tensor):
            save_dict[key] = save_dict[key].cpu().numpy().squeeze(0)

    save_for_viser(save_dict, scene_name, output_path, inverse_extrinsic=False)

    torch.cuda.synchronize()  # make sure the event timestamps are set
    duration = start_ev.elapsed_time(end_ev)
    gpu_mem_usage = torch.cuda.max_memory_allocated()

    summary_text = f"""
    Summary:
        Inference sec: {duration / 1000}
        Peak GPU memory usage (GB): {gpu_mem_usage / (1024 ** 3)} 
    """
    print(summary_text)

    # save_cache_to_viser(model.cache_dir, scene_name, output_path, overlap)


def run_dynamic_scene(args):
    data_path = args.data_path
    scene_name = data_path.split('/')[-2] if not args.scene_name else args.scene_name

    img_names = os.listdir(data_path)
    img_names = [os.path.join(data_path, name) for name in img_names if name.endswith(('.png', '.jpg', '.jpeg'))][
                ::args.sample_interval]
    print(f'Found {len(img_names)} images.')
    run_model(img_names, scene_name, args.output_path)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    model = load_model(args)

    model.eval()
    run_dynamic_scene(args)
