import random
import h5py
import scipy
import numpy as np
import cv2
import torch
import io
from PIL import Image
from torch.utils.data import Dataset
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
    
def center_crop_to_desired_hw(img, desired_height, desired_width):
    h, w, _ = img.shape
    y1 = (h - desired_height) // 2
    y2 = y1 + desired_height
    x1 = (w - desired_width) // 2
    x2 = x1 + desired_width
    img = img[y1:y2, x1:x2]
    return img


def read_image_from_bytes(b):
    dset_read_np = np.array(b)
    img_res = Image.open(io.BytesIO(dset_read_np))
    img_res = np.array(img_res)
    return img_res


def central_crop_img_arr(img, d=256):
    h, w, c = img.shape
    assert min(h, w) == d 
    s = d 
    oh_resid = (h - s) % 2
    ow_resid = (w - s) % 2
    oh = (h - s) // 2
    ow = (w - s) // 2
    img = img[oh : h - oh - oh_resid, ow : w - ow - ow_resid]
    assert img.shape == (d, d, c), img.shape
    return img


def convert_raw_extrinsics_to_Twc(raw_data):
    """
    helper function that convert raw extrinsics (6d pose) to world2cam transformation matrix (Twc)
    """
    raw_data = np.array(raw_data)
    pos = raw_data[0:3]
    rot_mat = scipy.spatial.transform.Rotation.from_euler('xyz', raw_data[3:6]).as_matrix()
    extrinsics = np.eye(4)
    extrinsics[:3,:3] = rot_mat
    extrinsics[:3,3] = pos
    # invert the matrix to represent standard definition of extrinsics: from world to cam
    # OpenCV -> OpenGL
    extrinsics[:, 1] *= -1
    extrinsics[:, 2] *= -1

    extrinsics = np.linalg.inv(extrinsics)
    return extrinsics


class IterableDroidDataset(torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.this = DroidDataset(*args, **kwargs)
        self.len = len(self.this)

    def __iter__(self):
        while True:
            yield self.this.__getitem__(np.random.randint(0, self.len))


class DroidDataset(Dataset):

    VIEWS_PER_SAMPLE = 4
    CAM_FOV = 68.0

    def __init__(self, file_paths, store_in_memory=True, compute_nearplane_quantile=False, load_single_batch_debug=False, **kwargs):
        """
        file_paths: a list of hdf5 file paths to load
        """
        self.file_paths = file_paths
        self.store_in_memory = store_in_memory
        self.compute_nearplane_quantile = compute_nearplane_quantile
        self.load_single_batch_debug = load_single_batch_debug

        if self.load_single_batch_debug:
            print("IMPORTANT! Using single batch debug mode. \n Only loading the first file.")
            self.file_paths = self.file_paths[:1]

        # open all files
        if store_in_memory:
            # use 'core' driver to load file into memory
            print("Loading droid data to memory...")
            self.files = [h5py.File(file_path, 'r', driver='core') for file_path in self.file_paths] 
        else:
            self.files = [h5py.File(file_path, 'r') for file_path in self.file_paths]
        self._compute_dataset_info()

    def _get_num_valid_samples(self, file):
        sample_num = 0
        while sample_num < 1000000:
            if not f"/sample_{sample_num}" in file:
                break
            sample_num += 1
        # now sample_num represents the highest index that does _not_ exist.
        return sample_num

    def __len__(self):
        return self.dset_len

    def _compute_dataset_info(self):
        self.samples_per_file = [self._get_num_valid_samples(f) for f in self.files]
        self.num_samples = sum(self.samples_per_file) 
        self.cumsum_samples = np.cumsum(self.samples_per_file)
        self.pair_idx_to_image_indices = list(permutations(range(self.VIEWS_PER_SAMPLE), 2))
        self.num_pairs_per_sample = len(self.pair_idx_to_image_indices)
        assert self.num_pairs_per_sample == self.VIEWS_PER_SAMPLE * (self.VIEWS_PER_SAMPLE - 1)
        self.dset_len = self.num_samples * self.num_pairs_per_sample

        if self.load_single_batch_debug:
            self.dset_len = 64

    def _get_data_for_key(self, group, key, prefix="cond"):
        image = read_image_from_bytes(group[key]) 
        image = image / 255.
        height, width, _ = image.shape
        desired_height, desired_width = height - height % 32, width - width % 32
        image = center_crop_to_desired_hw(image, desired_height, desired_width) # match depth width

        depth = group[f"{key}_depth"][()]
        depth = depth.astype(np.float32)
    
        # make shortest side 256
        h, w = image.shape[:2]
        if h < w:
            new_h, new_w = 256, int(256 * w / h)
        else:
            new_h, new_w = int(256 * h / w), 256
  
        image = cv2.resize(image, (new_w, new_h))
        depth = cv2.resize(depth, (new_w, new_h))

        # center crop
        image = central_crop_img_arr(image)
        depth = central_crop_img_arr(depth[..., None])

   

        image = image * 2 - 1 # convert to -1, 1
        image = image.astype(np.float32)
        
        world2cam = convert_raw_extrinsics_to_Twc(group[f"{key}_extrinsics"])
        return { 
            f"image_{prefix}": image,
            f"depth_{prefix}": depth,
            f"depth_{prefix}_filled": None,
            f"{prefix}_cam2world": np.linalg.inv(world2cam),
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
        sample_idx = idx // self.num_pairs_per_sample # the idx of the sample that we will grab 
        selected_pair_within_sample = idx % self.num_pairs_per_sample
        file_idx = 0
        while sample_idx >= self.cumsum_samples[file_idx]:
            file_idx += 1
        file = self.files[file_idx]
        sample_idx_in_file = sample_idx - (self.cumsum_samples[file_idx - 1] if file_idx > 0 else 0)
        sample = file[f"/sample_{sample_idx_in_file}"]
        cam_serials = [sample.attrs["ext1_cam_serial"], sample.attrs["ext2_cam_serial"]]
        cam_idx_to_dict_key = [f"{cam_serials[0]}_left", f"{cam_serials[0]}_right", 
                               f"{cam_serials[1]}_left", f"{cam_serials[1]}_right"]
        image_indices = self.pair_idx_to_image_indices[selected_pair_within_sample]
        cond_key, target_key = cam_idx_to_dict_key[image_indices[0]], cam_idx_to_dict_key[image_indices[1]]
        batch_struct = dict()
        for key, prefix in zip([cond_key, target_key], ["cond", "target"]):
            key_data_dict = self._get_data_for_key(sample, key, prefix)
            batch_struct.update(key_data_dict)

        world2cams = np.stack((np.linalg.inv(batch_struct["cond_cam2world"]), np.linalg.inv(batch_struct["target_cam2world"])))

        world_data = get_world_data(worldtocams=world2cams[:, :3])

        batch_struct["fov_deg"] = self.CAM_FOV

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
        batch_struct["fov_deg"] = self.CAM_FOV
        batch_struct["scale_adjustment"] = 1.0 # TODO
        batch_struct["nearplane_quantile"] = nearplane_quantile
        batch_struct["depth_cond_quantile25"] = None
        batch_struct["cond_elevation_deg"] = world_data['elevation_deg'][0]
        assert set(batch_struct.keys()) == set(webdataset_base.BATCH_STRUCT_KEYS), "batch struct does not have exactly keys needed"

        return webdataset_base.batch_struct_to_tuple(batch_struct)
        
        


                

        


        

