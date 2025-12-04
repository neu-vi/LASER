import os.path as osp
import glob

import cv2
import torchvision.transforms as tvf

from .base_dataset import BaseDataset
from .dataset_util import *
from utils.load_fn import load_and_preprocess_images

to_tensor = tvf.ToTensor()


class Waymo(BaseDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(
            self,
            WAYMO_DIR,
            seq_list=None,
            **kwargs
    ):
        super().__init__()
        self.WAYMO_DIR = WAYMO_DIR
        if WAYMO_DIR == None:
            raise NotImplementedError
        print(f"WAYMO_DIR is {WAYMO_DIR}")

        if seq_list is None:
            self.seq_list = os.listdir(self.WAYMO_DIR)
            self.seq_list = [seq for seq in self.seq_list if 'segment-' in seq]
            self.seq_list = sorted(self.seq_list)
        else:
            self.seq_list = [f'segment-{seq}_with_camera_labels.tfrecord' for seq in seq_list]

        self.seq_list_len = len(self.seq_list)
        self.img_size = 518

    def get_data(
            self,
            seq_index: int = None,
            sequence_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        if sequence_name is None:
            sequence_name = self.seq_list[seq_index]

        all_files = os.listdir(os.path.join(self.WAYMO_DIR, sequence_name))
        # rgb/depth/intrinsics - 5 cameras
        num_images = len(all_files) // 3 // 5

        if ids is None:
            ids = np.arange(num_images).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        # target_image_shape = self.get_target_shape(aspect_ratio)
        target_image_shape = np.asarray(
            load_and_preprocess_images(sorted(glob.glob(os.path.join(self.WAYMO_DIR, sequence_name, "*_1.jpg")))).shape[
            -2:])

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        first_frame_extrinsic_inv = None
        for i, id in enumerate(ids):
            rgb_path = os.path.join(self.WAYMO_DIR, sequence_name, f'{id:05d}_1.jpg')
            depth_path = os.path.join(self.WAYMO_DIR, sequence_name, f'{id:05d}_1.exr')

            image = read_image_cv2(rgb_path)
            camera_params = np.load(osp.join(self.WAYMO_DIR, sequence_name, f'{id:05d}_1.npz'))
            intri_opencv = camera_params['intrinsics']
            extri_opencv = camera_params['cam2world']  # camera_to_world
            extri_opencv = np.linalg.inv(extri_opencv)  # world_to_cam

            if i == 0:
                # extri_opencv: camA2World
                first_frame_extrinsic_inv = np.linalg.inv(extri_opencv)
            # camB2world @ inv(CamA2World) = CamB2CamA
            # camA is world coordinate
            extri_opencv = extri_opencv @ first_frame_extrinsic_inv

            original_size = np.array(image.shape[:2])

            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=rgb_path,
            )

            images.append(to_tensor(image))
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(rgb_path)
            original_sizes.append(original_size)

        batch = {
            "seq_id": sequence_name,
            "seq_len": len(extrinsics),
            "ind": torch.tensor(ids),
            'image_paths': image_paths,
            "images": torch.stack(images, dim=0),
            'pointclouds': np.stack(world_points, axis=0),
            'valid_mask': np.stack(depths, axis=0) > 1e-4,
            # "depths": depths,
            # "extrinsics": extrinsics,
            # "intrinsics": intrinsics,
            # "cam_points": cam_points,
            # "world_points": world_points,
            # "point_masks": point_masks,
            # "original_sizes": original_sizes,
        }

        return batch
