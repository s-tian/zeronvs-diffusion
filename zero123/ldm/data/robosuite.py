import random
import h5py
import scipy
import numpy as np
import cv2
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from itertools import permutations

from ldm.data import webdataset_base
from ldm.data import common


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def get_world_data(*, worldtocams):
    cams = worldtocams
    assert cams.ndim == 3
    assert cams.shape[1:] == (3, 4)
    h = np.array([0, 0, 0, 1])[None, None]
    h = np.broadcast_to(h, (cams.shape[0], h.shape[1], h.shape[2]))
    cams = np.concatenate([cams, h], axis=1)
    camtoworld = np.linalg.inv(cams)

    focus_pt = focus_point_fn(camtoworld)[None]

    locations = camtoworld[:, :3, -1]
    assert locations.shape == (cams.shape[0], 3)
    center = locations.mean(axis=0, keepdims=True)
    assert focus_pt.shape == center.shape, (focus_pt.shape, center.shape)
    scene_radius = np.linalg.norm(locations - center, axis=1).mean()
    scene_radius_focus_pt = np.linalg.norm(locations - focus_pt, axis=1).mean()
    cams = cams[:, :3]

    camera_loc = camtoworld[:, :3, -1]
    center_of_mass = camera_loc.mean(axis=0, keepdims=True)

    focus_pt_to_center_of_mass = np.linalg.norm(focus_pt - center_of_mass)
    camera_to_center_of_mass = np.linalg.norm((camera_loc - center_of_mass), axis=-1)
    elevation = np.arctan2(focus_pt_to_center_of_mass, camera_to_center_of_mass)
    elevation_deg = elevation * 180 / np.pi

    mean_elevation_deg = elevation_deg.mean()

    return {
        "center": center,
        "focus_pt": focus_pt,
        "scene_radius_focus_pt": scene_radius_focus_pt,
        "scene_radius": scene_radius,
        "camtoworld": camtoworld,
        "elevation_deg": elevation_deg,
        "mean_elevation_deg": mean_elevation_deg,
        "scale_adjustment": 1.0,
    }
    
def read_image_from_bytes(b):
    dset_read_np = np.array(b)
    img_res = Image.open(io.BytesIO(dset_read_np))
    img_res = np.array(img_res)
    return img_res

class IterableRobosuiteDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_paths, *args, **kwargs):
        super().__init__()
        single_datasets = []
        for file_path in file_paths:
            single_dataset = RobosuiteDataset(file_path, *args, **kwargs)
            single_datasets.append(single_dataset) 
        self.this = ConcatDataset(single_datasets)
        self.len = len(self.this)

    def __iter__(self):
        while True:
            yield self.this.__getitem__(np.random.randint(0, self.len))

class RobosuiteDataset(Dataset):

    VIEWS_PER_SAMPLE = 10 
    # a dataset that loads a robomimic dataset from a single hdf5 file 

    def __init__(self, file_path, store_in_memory=True, compute_nearplane_quantile=False, load_single_batch_debug=False, **kwargs):
        """
        file_paths: a list of hdf5 file paths to load
        """
        self.file_path = file_path
        self.store_in_memory = store_in_memory
        self.compute_nearplane_quantile = compute_nearplane_quantile
        self.load_single_batch_debug = load_single_batch_debug

        if self.load_single_batch_debug:
            print("IMPORTANT! Using single batch debug mode. \n Only loading the first file.")
            self.file_paths = self.file_paths[:1]

        # open the file 
        if store_in_memory:
            # use 'core' driver to load file into memory
            print(f"Loading robosuite data at path {self.file_path} to memory...")
            self.file = h5py.File(self.file_path, 'r', driver='core')
        else:
            self.file = h5py.File(self.file_path, 'r') 
        self._compute_dataset_info()
    
    def _get_num_demos(self, file):
        sample_idx = 0
        while sample_idx < 1000000:
            if not f"data/demo_{sample_idx}" in file:
                break
            sample_idx += 1 
        # now sample_num represents the highest index that does _not_ exist.
        return sample_idx 

    def _get_num_samples(self, file, demo_idx):
        return len(file[f"data/demo_{demo_idx}/dones"])
        
    def __len__(self):
        return self.dset_len

    def _compute_dataset_info(self):
        self.num_demos = self._get_num_demos(self.file)
        self.samples_per_demo = [self._get_num_samples(self.file, idx) for idx in range(self.num_demos)]
        print(self.samples_per_demo)
        self.num_samples = sum(self.samples_per_demo) 
        self.cumsum_samples = np.cumsum(self.samples_per_demo)
        self.pair_idx_to_image_indices = list(permutations(range(self.VIEWS_PER_SAMPLE), 2))
        self.num_pairs_per_sample = len(self.pair_idx_to_image_indices)
        assert self.num_pairs_per_sample == self.VIEWS_PER_SAMPLE * (self.VIEWS_PER_SAMPLE - 1)
        self.dset_len = self.num_samples * self.num_pairs_per_sample

        if self.load_single_batch_debug:
            self.dset_len = 64

    def _get_data_for_key(self, group, key, time_idx, prefix="cond"):
        # key is something like agentview_5
        image = read_image_from_bytes(group[f"{key}_image"][time_idx]) 
        image = image / 255.
        height, width, _ = image.shape

        depth = group[f"{key}_depth"][time_idx]
        depth = depth.astype(np.float32)
       
        image = image * 2 - 1 # convert to -1, 1
        image = image.astype(np.float32)
        
        cam2world = group[f"{key}_extrinsics"][time_idx]
        # OpenCV -> OpenGL
        cam2world[:, 1] *= -1
        cam2world[:, 2] *= -1

        intrinsics = group[f"{key}_intrinsics"][time_idx]

        return { 
            f"image_{prefix}": image,
            f"depth_{prefix}": depth,
            f"depth_{prefix}_filled": None,
            f"{prefix}_cam2world": cam2world,
            f"{prefix}_intrinsics": intrinsics,
        }

    def __getitem__(self, idx):
        """
        batch_struct has the following required keys:
            "image_target", DONE
            "image_cond",   DONE
            "depth_target", DONE
            "depth_target_filled", DONE
            "depth_cond", DONE
            "depth_cond_filled", DONE
            "uid", DONE
            "pair_uid", DONE
            "T", DONE
            "target_cam2world", DONE
            "cond_cam2world", DONE
            "center", DONE
            "focus_pt", DONE
            "scene_radius_focus_pt", DONE
            "scene_radius", DONE
            "fov_deg", DONE
            "scale_adjustment", TODO this is just a scaling factor that is applied to the depth map
            "nearplane_quantile", DONE
            "depth_cond_quantile25", DONE
            "cond_elevation_deg", DONE
        """
        # print("idx", idx)
        sample_idx = idx // self.num_pairs_per_sample # the idx of the sample that we will grab 
        # print("sample idx", sample_idx)
        selected_pair_within_sample = idx % self.num_pairs_per_sample
        # print("selected pair", selected_pair_within_sample)
        file_idx = 0
        while sample_idx >= self.cumsum_samples[file_idx]:
            file_idx += 1
        # print("File idx", file_idx)
        file = self.file[f"data/demo_{file_idx}/obs"]
        sample_idx_in_file = sample_idx - (self.cumsum_samples[file_idx - 1] if file_idx > 0 else 0)
        # print("sample idx in file", sample_idx_in_file)
        cam_idx_to_dict_key = lambda x: "agentview_" + str(x)
        image_indices = self.pair_idx_to_image_indices[selected_pair_within_sample]
        cond_key, target_key = cam_idx_to_dict_key(image_indices[0]), cam_idx_to_dict_key(image_indices[1])
        batch_struct = dict()
        for key, prefix in zip([cond_key, target_key], ["cond", "target"]):
            key_data_dict = self._get_data_for_key(file, key, sample_idx_in_file, prefix)
            batch_struct.update(key_data_dict)

        world2cams = np.stack((np.linalg.inv(batch_struct["cond_cam2world"]), np.linalg.inv(batch_struct["target_cam2world"])))
        # print(world2cams)
        # print(world2cams.shape)
        # print(world2cams[:, :3])
        world_data = get_world_data(worldtocams=world2cams[:, :3])
        fx = batch_struct["cond_intrinsics"][0,0]
        batch_struct["fov_deg"] = FOV = np.rad2deg(2 * np.arctan2(256, 2 * fx))

        # pop intrinsics as they are not used and batch_struct keys need to match exactly
        batch_struct.pop("cond_intrinsics")
        batch_struct.pop("target_intrinsics")

        batch_struct["T"] = T = common.get_T(
            np.linalg.inv(batch_struct["target_cam2world"])[:3],
            np.linalg.inv(batch_struct["cond_cam2world"])[:3],
            to_torch=False,
        ).astype(np.float32)

        if self.compute_nearplane_quantile: 
            depths = [
                batch_struct[f"depth_{key}"] * world_data["scale_adjustment"]
                for key in ["cond", "target"]
            ]
            masked_depths = [depth[depth != 0] for depth in depths]
            quantiles = [
                np.quantile(depth, 0.05) if len(depth) > 0 else None
                for depth in masked_depths
            ]
            
            if all(quantile is None for quantile in quantiles):
                print("got scene with no points! this should not happen often!")
                return
            else:
                nonempty_quantiles = np.array(
                    [quantile for quantile in quantiles if quantile is not None]
                )
                nearplane_quantile = np.quantile(nonempty_quantiles, 0.1) 
        else:
            nearplane_quantile = None
        
        batch_struct["uid"] = f"{file_idx}_{sample_idx_in_file}" # scene uid
        batch_struct["pair_uid"] = f"{file_idx}_{sample_idx_in_file}_{selected_pair_within_sample}" # scene uid with pair idx
        batch_struct["center"] = world_data["center"]
        batch_struct["focus_pt"] = world_data["focus_pt"]
        batch_struct["scene_radius"] = world_data["scene_radius"]
        batch_struct["scene_radius_focus_pt"] = world_data["scene_radius_focus_pt"]
        batch_struct["fov_deg"] = FOV
        batch_struct["scale_adjustment"] = 1.0 # TODO
        batch_struct["nearplane_quantile"] = nearplane_quantile
        batch_struct["depth_cond_quantile25"] = None
        batch_struct["cond_elevation_deg"] = world_data['elevation_deg'][0]
        assert set(batch_struct.keys()) == set(webdataset_base.BATCH_STRUCT_KEYS), "batch struct does not have exactly keys needed"

        return webdataset_base.batch_struct_to_tuple(batch_struct)
        
        


                

        


        

