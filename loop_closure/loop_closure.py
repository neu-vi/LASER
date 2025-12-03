import argparse
import torch
from pathlib import Path

from .loop_model import LoopDetector
from .utils.sim3loop import Sim3LoopOptimizer
from .utils.sim3utils import *

from utils.load_fn import load_and_preprocess_images
from pi3.models.pi3 import Pi3
from inference_engine import StreamingWindowEngine
from inference_engine.inference_utils import register_adjacent_windows

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def get_args_parser():
    parser = argparse.ArgumentParser('Post process loop closure', add_help=False)
    parser.add_argument('--config_path', default=None, type=str, help='loop closure config')
    parser.add_argument('--data_path', type=str, help='sequence data path')
    parser.add_argument('--cache_path', default='./cache', type=str,
                        help='inference cache path')
    parser.add_argument('--output_path', default='./cache_lc', type=str,
                        help='loop closure cache path')
    parser.add_argument('--window_size', default=20, type=int, help='sliding window size')
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')

    return parser


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    return result


class LoopClosureEngine:
    def __init__(
            self,
            config,
            image_dir,
            output_dir,
            pi3_model,
            window_size,
            overlap,
            top_conf_percentile=0.5
    ):
        self.config = config

        self.pi3_model = pi3_model
        self.window_size = window_size
        self.overlap = overlap
        self.top_conf_percentile = top_conf_percentile

        self.img_dir = image_dir
        self.img_list = None

        self.loop_detector = LoopDetector(
            image_dir=image_dir,
            output=output_dir,
            config=config
        )

        self.chunk_indices = None
        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.loop_list = []  # e.g. [(1584, 139), ...]
        self.loop_optimizer = Sim3LoopOptimizer(config)
        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]
        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]
        self.loop_predict_list = []

    def get_loop_pairs(self):
        self.loop_detector.run()
        self.loop_list = self.loop_detector.get_loop_list()
        del self.loop_detector
        torch.cuda.empty_cache()

    def process_single_chunk(self, range_1, range_2=None):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        images = load_and_preprocess_images(chunk_image_paths).to(device)

        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.pi3_model(images)
        torch.cuda.empty_cache()

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().squeeze(0)

        conf_thre = torch.quantile(predictions['conf'], self.top_conf_percentile, interpolation='nearest')
        predictions['mask'] = predictions['conf'] >= conf_thre

        return predictions

    def process_loops(self, raw_predictions):
        step = self.window_size - self.overlap
        num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
        self.chunk_indices = []
        for i in range(num_chunks):
            start_idx = i * step
            end_idx = min(start_idx + self.window_size, len(self.img_list))
            self.chunk_indices.append((start_idx, end_idx))

        print('Loop SIM(3) estimating...')
        loop_results = process_loop_list(self.chunk_indices,
                                         self.loop_list,
                                         half_window=self.config['Model']['loop_chunk_size'] // 2)
        loop_results = remove_duplicates(loop_results)
        for item in loop_results:
            single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3])
            self.loop_predict_list.append((item, single_chunk_predictions))

        for item in self.loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop = item[1]['local_points'][:chunk_a_range[1] - chunk_a_range[0]]
            cam_pose_loop = item[1]['camera_poses'][:chunk_a_range[1] - chunk_a_range[0]]
            conf_mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
            chunk_data_a = raw_predictions[chunk_idx_a]
            point_map_a = chunk_data_a['local_points'][chunk_a_rela_begin:chunk_a_rela_end]
            cam_pose_a = chunk_data_a['camera_poses'][chunk_a_rela_begin:chunk_a_rela_end]
            conf_mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]

            s_a, R_a, t_a = register_adjacent_windows(
                point_map_a,
                point_map_loop,
                cam_pose_a,
                cam_pose_loop,
                conf_mask_loop & conf_mask_a
            )

            point_map_loop = item[1]['local_points'][-chunk_b_range[1] + chunk_b_range[0]:]
            cam_pose_loop = item[1]['camera_poses'][-chunk_b_range[1] + chunk_b_range[0]:]
            conf_mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]

            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
            chunk_data_b = raw_predictions[chunk_idx_b]
            point_map_b = chunk_data_b['local_points'][chunk_b_rela_begin:chunk_b_rela_end]
            cam_pose_b = chunk_data_b['camera_poses'][chunk_b_rela_begin:chunk_b_rela_end]
            conf_mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]

            s_b, R_b, t_b = register_adjacent_windows(
                point_map_b,
                point_map_loop,
                cam_pose_b,
                cam_pose_loop,
                conf_mask_loop & conf_mask_b
            )

            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            # not inverse
            self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        # not inverse
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)

    def run(self, raw_predictions):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))

        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        self.get_loop_pairs()
        for pred in raw_predictions[1:]:
            self.sim3_list.append(pred['sim3'])

        self.process_loops(raw_predictions)
        return self.sim3_list


if __name__ == '__main__':
    from .utils.config_utils import load_config

    args = get_args_parser()
    args = args.parse_args()
    pi3_model = Pi3.from_pretrained("yyfz233/Pi3").to(device)

    config = load_config(args.config_path)
    loop_closure = LoopClosureEngine(
        config,
        args.data_path,
        args.output_path,
        pi3_model,
        args.window_size,
        args.overlap,
    )

    cache_files = sorted(glob.glob(str(Path(args.cache_path) / 'window_cache_*.pt')),
                         key=lambda p: int(p.split('_')[-1].split('.')[0]))
    raw_predictions = [StreamingWindowEngine.parse_cache_file(cache_fname) for cache_fname in cache_files]
    sim3_list_lc = loop_closure.run(raw_predictions)
    sim3_list_lc.insert(0, raw_predictions[0]['sim3'])

    for idx, (pred, sim3_lc) in enumerate(zip(raw_predictions, sim3_list_lc)):
        pred['sim3'] = sim3_lc
        torch.save(pred, f'{args.output_path}/window_cache_{idx}.pt')
