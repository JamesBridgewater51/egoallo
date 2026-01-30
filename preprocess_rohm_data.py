"""Convert RoHM-formatted AMASS data (npy) to EgoAllo/HuMoR-style npz format.

Formatted to match third_party/egoallo/0a_preprocess_training_data.py but reads
from the directory structure defined in data_sets/dataset_amass_like.py.
"""

import dataclasses
import os
import time
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from loguru import logger as guru
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Reuse imports from existing script
from egoallo.preprocessing.body_model import (
    KEYPT_VERTS,
    SMPL_JOINTS,
    BodyModel,
    reflect_pose_aa,
    reflect_root_trajectory,
    run_smpl,
)
from egoallo.preprocessing.geometry import convert_rotation, joints_global_to_local
from egoallo.preprocessing.util import move_to

# Re-define constants or import if possible (AMASS_SPLITS is useful for dataset iteration)
AMASS_SPLITS = {
    "train": [
        "ACCAD",
        "BMLhandball",
        "BMLmovi",
        "BioMotionLab_NTroje",
        "CMU",
        "DFaust_67",
        "DanceDB",
        "EKUT",
        "Eyes_Japan_Dataset",
        "KIT",
        "MPI_Limits",
        "TCD_handMocap",
        "TotalCapture",
    ],
    "val": [
        "HumanEva",
        "MPI_HDM05",
        "SFU",
        "MPI_mosh",
    ],
    "test": [
        "Transitions_mocap",
        "SSM_synced",
    ],
}
AMASS_SPLITS["all"] = AMASS_SPLITS["train"] + AMASS_SPLITS["val"] + AMASS_SPLITS["test"]

# Nymeria Split Config
NYMERIA_SPLIT_ROOT = "/home/minghao/src/robotflow/RoHM/datasets/Nymeria_smplx_preprocessed/nymeria_splits"


def load_neutral_beta_conversion(gender: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load gender to neutral beta conversion matrix."""
    assert gender in ["female", "male"]
    # Assuming running from root or adjusting path relative to where script is called
    path = Path("./data/smplh_gender_conversion") / f"{gender}_to_neutral.npz"
    if not path.exists():
         # Fallback try relative to script location if needed, 
         # but for now assume CWD is project root as per original script
         pass
    data = np.load(str(path))
    return data["A"], data["b"]


def convert_gender_neutral_beta(
    beta: np.ndarray, A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    :param beta (*, B)
    :param A (B, B)
    :param b (B)
    beta_neutral = A @ beta_gender + b
    """
    *dims, B = beta.shape
    A = A.reshape((*(1,) * len(dims), B, B))
    b = b.reshape((*(1,) * len(dims), B))
    return np.einsum("...ij,...j->...i", A, beta) + b


# Copied from 0a_preprocess_training_data.py
def determine_floor_height_and_contacts(
    body_joint_seq,
    fps,
    vis=False,
    floor_vel_thresh=0.005,
    floor_height_offset=0.01,
    contact_vel_thresh=0.005,  # 0.015
    contact_toe_height_thresh=0.04,  # if static toe above this height
    contact_ankle_height_thresh=0.08,
    terrain_height_thresh=0.04,
    root_height_thresh=0.04,
    cluster_size_thresh=0.25,
    discard_terrain_seqs=False,  # throw away person steps onto objects (determined by a heuristic)
):
    """
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    """
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS["hips"], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS["leftToeBase"], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS["rightToeBase"], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < floor_vel_thresh]
    left_static_inds = all_inds[left_toe_vel < floor_vel_thresh]
    right_static_foot_heights = right_toe_heights[right_toe_vel < floor_vel_thresh]
    right_static_inds = all_inds[right_toe_vel < floor_vel_thresh]

    all_static_foot_heights = np.append(
        left_static_foot_heights, right_static_foot_heights
    )
    all_static_inds = np.append(left_static_inds, right_static_inds)

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(
            all_static_foot_heights.reshape(-1, 1)
        )
        all_labels = np.unique(clustering.labels_)
        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )  # inds in the original sequence that correspond to this cluster
            
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median
        offset_floor_height = (
            floor_height - floor_height_offset
        )  # toe joint is actually inside foot mesh a bit

        if discard_terrain_seqs:
            for cluster_root_height, cluster_height, cluster_size in zip(
                cluster_root_heights, cluster_heights, cluster_sizes
            ):
                root_above_thresh = cluster_root_height > (
                    min_root_median + root_height_thresh
                )
                toe_above_thresh = cluster_height > (min_median + terrain_height_thresh)
                cluster_size_above_thresh = cluster_size > int(
                    cluster_size_thresh * fps
                )
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    # print("DISCARDING sequence based on terrain interaction!")
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS["leftFoot"], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS["rightFoot"], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < contact_vel_thresh
    right_heel_contact = right_heel_vel < contact_vel_thresh
    left_toe_contact = left_toe_vel < contact_vel_thresh
    right_toe_contact = right_toe_vel < contact_vel_thresh

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights = left_toe_heights - floor_height
    right_toe_heights = right_toe_heights - floor_height

    left_heel_contact = np.logical_and(
        left_heel_contact, left_heel_heights < contact_ankle_height_thresh
    )
    right_heel_contact = np.logical_and(
        right_heel_contact, right_heel_heights < contact_ankle_height_thresh
    )
    left_toe_contact = np.logical_and(
        left_toe_contact, left_toe_heights < contact_toe_height_thresh
    )
    right_toe_contact = np.logical_and(
        right_toe_contact, right_toe_heights < contact_toe_height_thresh
    )

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:, SMPL_JOINTS["leftFoot"]] = left_heel_contact
    contacts[:, SMPL_JOINTS["leftToeBase"]] = left_toe_contact
    contacts[:, SMPL_JOINTS["rightFoot"]] = right_heel_contact
    contacts[:, SMPL_JOINTS["rightToeBase"]] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(
        body_joint_seq,
        "leftHand",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    right_hand_contact = detect_joint_contact(
        body_joint_seq,
        "rightHand",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    contacts[:, SMPL_JOINTS["leftHand"]] = left_hand_contact
    contacts[:, SMPL_JOINTS["rightHand"]] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(
        body_joint_seq,
        "leftLeg",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    right_knee_contact = detect_joint_contact(
        body_joint_seq,
        "rightLeg",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    contacts[:, SMPL_JOINTS["leftLeg"]] = left_knee_contact
    contacts[:, SMPL_JOINTS["rightLeg"]] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def detect_joint_contact(
    body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh
):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact


def compute_root_align_mats(root_orient):
    root_orient = torch.as_tensor(root_orient).reshape(-1, 3)
    # convert aa to matrices
    root_orient_mat = convert_rotation(root_orient, "aa", "mat").numpy()

    # rotate root so aligning local body right vector (-x) with world right vector (+x)
    #       with a rotation around the up axis (+z)
    # in body coordinates body x-axis is to the left
    body_right = -root_orient_mat[:, :, 0]
    world2aligned_mat, world2aligned_aa = compute_align_from_body_right(body_right)

    return world2aligned_mat


def compute_joint_align_mats(joint_seq):
    left_idx = SMPL_JOINTS["leftUpLeg"]
    right_idx = SMPL_JOINTS["rightUpLeg"]

    body_right = joint_seq[:, right_idx] - joint_seq[:, left_idx]
    body_right = body_right / np.linalg.norm(body_right, axis=1)[:, np.newaxis]

    world2aligned_mat, world2aligned_aa = compute_align_from_body_right(body_right)

    return world2aligned_mat


def compute_align_from_body_right(body_right):
    world2aligned_angle = np.arccos(
        body_right[:, 0] / (np.linalg.norm(body_right[:, :2], axis=1) + 1e-8)
    )  # project to world x axis, and compute angle
    body_right[:, 2] = 0.0
    world2aligned_axis = np.cross(body_right, np.array([[1.0, 0.0, 0.0]]))

    world2aligned_aa = (
        world2aligned_axis
        / (np.linalg.norm(world2aligned_axis, axis=1)[:, np.newaxis] + 1e-8)
    ) * world2aligned_angle[:, np.newaxis]

    world2aligned_mat = convert_rotation(
        torch.as_tensor(world2aligned_aa).reshape(-1, 3), "aa", "mat"
    ).numpy()

    return world2aligned_mat, world2aligned_aa


def estimate_velocity(data_seq, h):
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2 * h)
    return data_vel_seq


def estimate_angular_velocity(rot_seq, h):
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w


def load_rohm_seq_data(joints_path: str, smpl_path: str):
    """
    Load data from RoHM-formatted .npy files.
    
    Format ref: dataset_amass_like.py
    pose_data: [T, 25, 3] (Joints)
    smpl_data: [T, 178] (SMPL params)
    
    Mapping from 178 dim vector:
    0:3   -> global_orient
    3:6   -> trans
    6:16  -> betas
    16:79 -> body_pose (21 joints)
    """
    # guru.info(f"Loading from {joints_path} & {smpl_path}")
    
    seq_joints = np.load(joints_path) # [T, 25, 3]
    seq_smplx = np.load(smpl_path)    # [T, 178]
    
    num_frames = seq_smplx.shape[0]
    
    # Defaults per requirements
    gender = "male"
    fps = 30.0
    
    # Extract SMPL params
    # RoHM smplx data layout from dataset_amass_like.py
    root_orient = seq_smplx[:, 0:3]
    trans = seq_smplx[:, 3:6]
    betas = seq_smplx[:, 6:16]
    # Tile 10-dim betas to 16-dims
    betas = np.concatenate([betas, np.zeros((num_frames, 6))], axis=1)
    pose_body = seq_smplx[:, 16:79] # 63 dims (21 joints)
    
    # Construct zero-padded hands
    # SMPL usually expects more joints. 0a uses 66: for hands.
    # 21 body joints * 3 = 63. 
    # If we need to satisfy SMPL-H or similar which expects 156 total pose dims (3 root + 63 body + 90 hand)
    # We will create zero hand poses.
    # The requirement says: "for last left/right wrist, default to 0"
    pose_hand = np.zeros((num_frames, 90), dtype=np.float32) 
    
    model_vars = {
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "betas": betas,
    }
    
    meta = {"fps": fps, "gender": gender, "num_frames": num_frames}
    
    # Optional debug info
    # guru.info(f"meta {meta}")
    # guru.info(f"model var shapes {str({k: v.shape for k, v in model_vars.items()})}")
    
    return model_vars, meta


def run_batch_smpl(
    body_model: BodyModel,
    device: torch.device,
    num_total: int,
    batch_size: int,
    return_verts: bool = True,
    **kwargs,
):
    var_dims = body_model.var_dims
    var_names = [name for name in kwargs if name in var_dims]
    model_vars = {
        name: torch.as_tensor(kwargs[name], dtype=torch.float32).reshape(
            -1, var_dims[name]
        )
        for name in var_names
    }
    fopts = {k: v for k, v in kwargs.items() if k not in var_names}

    batch_joints, batch_verts = [], []
    for sidx in range(0, num_total, batch_size):
        eidx = min(sidx + batch_size, num_total)
        batch_model_vars = move_to(
            {name: x[sidx:eidx].contiguous() for name, x in model_vars.items()}, device
        )
        with torch.no_grad():
            joints, verts, _ = run_smpl(
                body_model, return_verts=return_verts, **batch_model_vars, **fopts
            )
        batch_joints.append(joints.detach().cpu())
        if return_verts and verts is not None:
            batch_verts.append(verts.detach().cpu())

    joints_all = torch.cat(batch_joints, dim=0)
    verts_all = torch.cat(batch_verts, dim=0) if len(batch_verts) > 0 else None
    return joints_all, verts_all


def process_seq(
    joints_path: str,
    smpl_path: str,
    out_path: str,
    smplh_root: str,
    dev_id: int,
    beta_neutral: bool,
    reflect: bool = False,
    overwrite: bool = False,
    **kwargs,
):
    if not overwrite and os.path.isfile(out_path):
        # guru.info(f"{out_path} already exists, skipping.")
        return

    # guru.info(f"process to {out_path}")

    # Use custom loader for RoHM format
    model_vars, meta = load_rohm_seq_data(joints_path, smpl_path)

    if beta_neutral:  # get the gender neutral beta
        # guru.info("converting betas to gender neutral")
        try:
            A_beta, b_beta = load_neutral_beta_conversion(meta["gender"])
            model_vars["betas"] = convert_gender_neutral_beta(
                model_vars["betas"], A_beta, b_beta
            )
            meta["gender"] = "neutral"
        except Exception as e:
            guru.warning(f"Failed to load neutral beta conversion: {e}. Skipping conversion.")

    process_seq_data(
        model_vars, meta, out_path, dev_id, smplh_root, reflect=reflect, **kwargs
    )


def process_seq_data(
    model_vars: Dict,
    meta: Dict,
    out_path: str,
    dev_id: int,
    smplh_root: str,
    reflect: bool = False,
    split_frame_limit: int = 2000,
    discard_shorter_than: float = 1.0,  # seconds
    out_fps: int = 30,
    save_verts: bool = False,
    save_velocities: bool = True,  # save all parameter velocities available
):
    # guru.info(f"Processing seq with meta {meta}")
    start_t = time.time()

    gender = meta["gender"]
    src_fps = meta["fps"]
    num_frames = meta["num_frames"]

    # RoHM data is already processed/sliced effectively? 
    # Original script keeps middle 80%. 
    # Logic from 0a:
    # "only keep middle 80% of sequences to avoid redundanct static poses"
    sidx, eidx = int(0.1 * num_frames), int(0.9 * num_frames)
    num_frames = eidx - sidx
    for name, x in model_vars.items():
        model_vars[name] = x[sidx:eidx]
    # guru.info(str({k: v.shape for k, v in model_vars.items()}))

    # discard if shorter than threshold
    if num_frames < discard_shorter_than * src_fps:
        guru.info(f"Sequence shorter than {discard_shorter_than} s, discarding...")
        return

    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    device = (
        torch.device(f"cuda:{dev_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # <HACKS>
    # smplx tries to read shape properties, even when use_pca=False
    from smplx.utils import Struct

    Struct.hands_componentsl = np.zeros(100)  # type: ignore
    Struct.hands_componentsr = np.zeros(100)  # type: ignore
    Struct.hands_meanl = np.zeros(100)  # type: ignore
    Struct.hands_meanr = np.zeros(100)  # type: ignore

    # This defaults to 300, but we have 16 beta parameters. When
    # 16<300 the SMPL class will set num_betas to 10...
    from smplx import SMPLH

    assert SMPLH.SHAPE_SPACE_DIM in (300, 16)
    SMPLH.SHAPE_SPACE_DIM = 16
    # <HACKS>

    # Use SMPLH model
    # Note: RoHM data uses SMPLX body_pose (21 joints). 
    # Loading this into SMPLH might require careful mapping or it works if topology matches.
    # SMPL-H: Root(1) + Body(21) + Hands(15*2=30). 
    # Our model_vars has pose_body (21) and pose_hand (zeros).
    try:
        body_model = BodyModel(f"{smplh_root}/{gender}/model.npz", use_pca=False).to(device)
    except Exception as e:
        guru.error(f"Failed to load BodyModel from {smplh_root}/{gender}/model.npz: {e}")
        return

    model_vars = {k: torch.as_tensor(v).float() for k, v in model_vars.items()}
    if reflect:
        rot_og = model_vars["root_orient"]
        rot_re, model_vars["pose_body"] = reflect_pose_aa(
            rot_og, model_vars["pose_body"]
        )
        out = body_model.forward(betas=model_vars["betas"][:1].to(device))
        root_loc = out.Jtr[:, 0].cpu()  # type: ignore
        model_vars["root_orient"], model_vars["trans"] = reflect_root_trajectory(
            rot_og, model_vars["trans"], rot_re, root_loc
        )

    body_joint_seq, body_vtx_seq = run_batch_smpl(
        body_model,
        device,
        num_frames,
        split_frame_limit,
        return_verts=save_verts,
        **model_vars,
    )
    joints_glob = body_joint_seq[:, : len(SMPL_JOINTS), :]
    joint_seq = joints_glob.numpy()

    # guru.info(f"Recovered joints and verts {joint_seq.shape}")

    out_dict = model_vars.copy()
    out_dict["joints"] = joint_seq
    out_dict["joints_loc"], _ = joints_global_to_local(
        convert_rotation(model_vars["root_orient"], "aa", "mat"),
        model_vars["trans"],
        joints_glob,
    )

    if save_verts and body_vtx_seq is not None:
        out_dict["mojo_verts"] = body_vtx_seq[:, KEYPT_VERTS, :].numpy()

    # determine floor height and foot contacts
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(
        joint_seq, src_fps
    )

    if discard_seq:
        guru.info("Terrain interaction detected, discarding...")
        return

    # guru.info(f"Floor height: {floor_height}")
    # translate so floor is at z=0
    for name in ["trans", "joints", "mojo_verts"]:
        if name not in out_dict:
            continue
        out_dict[name][..., 2] -= floor_height

    # compute rotation to canonical frame (forward facing +y) for every frame
    world2aligned_rot = compute_root_align_mats(model_vars["root_orient"])

    out_dict.update(
        {
            "contacts": contacts,
            "floor_height": floor_height,
            "world2aligned_rot": world2aligned_rot,
        }
    )

    # estimate various velocities based on full frame rate
    #       with second order central differences before downsampling
    if save_velocities:
        h = 1.0 / src_fps
        lin_names = ["trans", "joints", "mojo_verts"]
        ang_names = ["root_orient", "pose_body"]
        cur_keys = lin_names + ang_names + ["contacts"]

        for name in lin_names:
            if name not in out_dict:
                continue
            out_dict[f"{name}_vel"] = estimate_velocity(out_dict[name], h)

        # root orient
        for name in ang_names:
            if name not in out_dict:
                continue
            rot_aa = (
                torch.as_tensor(out_dict[name]).reshape(num_frames, -1, 3).squeeze()
            )
            rot_mat = convert_rotation(rot_aa, "aa", "mat").numpy()
            out_dict[f"{name}_vel"] = estimate_angular_velocity(rot_mat, h)

        # joint up-axis angular velocity (need to compute joint frames first...)
        # need the joint transform at all steps to find the angular velocity
        joints_world2aligned_rot = compute_joint_align_mats(joint_seq)
        joint_orient_vel = -estimate_angular_velocity(joints_world2aligned_rot, h)
        # only need around z
        out_dict["joint_orient_vel"] = joint_orient_vel[:, 2]

        # throw out edge frames for other data so velocities are accurate
        for name in cur_keys:
            if name not in out_dict:
                continue
            out_dict[name] = out_dict[name][1:-1]
        num_frames = num_frames - 2

    # downsample frames
    fps_ratio = float(out_fps) / src_fps
    # guru.info(f"Downsamp ratio: {fps_ratio}")
    new_num_frames = int(fps_ratio * num_frames)
    # guru.info(f"Downsamp num frames: {new_num_frames}")
    downsamp_inds = np.linspace(0, num_frames - 1, num=new_num_frames, dtype=int)

    for k, v in out_dict.items():
        # print(k, type(v))
        if not isinstance(v, (torch.Tensor, np.ndarray)):
            continue
        if v.ndim >= 1:
            # print("downsampling", k)
            out_dict[k] = v[downsamp_inds]

    meta = {
        "fps": out_fps,
        "num_frames": new_num_frames,
        "gender": str(gender),
    }

    # guru.info(f"Seq process time: {time.time() - start_t} s")
    # guru.info(f"Saving data to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **meta, **out_dict)


@dataclasses.dataclass
class Config:
    data_root: str
    """Where the RoHM preprocessed dataset is stored (containing pose_data_fps_30)."""

    smplh_root: str = "./data/smplh"
    out_root: str = "./data/rohm_processed_for_egoallo/"
    devices: tuple[int, ...] = (0,)
    """CUDA devices. We use CPU if not available."""
    overwrite: bool = False
    
    process_nymeria: bool = False
    """Whether to process Nymeria dataset using its own splits."""
    nymeria_split_root: str = NYMERIA_SPLIT_ROOT


# Copied strict skip logic check
def check_skip(path_name: str) -> bool:
    """Copied conditions from https://github.com/davrempe/humor/blob/main/humor/scripts/cleanup_amass_data.py"""
    if "BioMotionLab_NTroje" in path_name and (
        "treadmill" in path_name or "normal_" in path_name
    ):
        return True
    if "MPI_HDM05" in path_name and "dg/HDM_dg_07-01" in path_name:
        return True
    return False


def get_dataset_files(data_root, dataset_name):
    """Retrieve matched joints and smpl files as per dataset_amass_like.py logic."""
    preprocessed_dataset_joints_dir = os.path.join(data_root, 'pose_data_fps_30')
    preprocessed_dataset_smpl_dir = os.path.join(data_root, 'smpl_data_fps_30')
    
    if dataset_name is not None:
         # Recursive glob
        seq_joints_paths = glob.glob(os.path.join(preprocessed_dataset_joints_dir, dataset_name, '**/*.npy'), recursive=True)
        seq_smpl_paths = glob.glob(os.path.join(preprocessed_dataset_smpl_dir, dataset_name, '**/*.npy'), recursive=True)
    else:
        # Not typically reached if we iterate datasets
        seq_joints_paths = glob.glob(os.path.join(preprocessed_dataset_joints_dir, '**/*.npy'), recursive=True)
        seq_smpl_paths = glob.glob(os.path.join(preprocessed_dataset_smpl_dir, '**/*.npy'), recursive=True)

    seq_joints_paths = sorted(seq_joints_paths)
    seq_smpl_paths = sorted(seq_smpl_paths)
    
    
    if len(seq_joints_paths) != len(seq_smpl_paths):
        guru.warning(f"Mismatch in file counts for {dataset_name}: {len(seq_joints_paths)} joints vs {len(seq_smpl_paths)} smpl files.")
        # Proceed with min length zip
    
    return list(zip(seq_joints_paths, seq_smpl_paths))

def get_nymeria_tasks(cfg: Config) -> List[Dict]:
    """Load Nymeria tasks based on split files."""
    tasks = []
    split_files = ["train.txt", "val.txt", "test.txt"]
    
    # Check if Nymeria folder exists in data_root
    nymeria_pose_root = os.path.join(cfg.data_root, 'pose_data_fps_30', 'Nymeria')
    nymeria_smpl_root = os.path.join(cfg.data_root, 'smpl_data_fps_30', 'Nymeria')
    
    if not os.path.exists(nymeria_pose_root):
        guru.warning(f"Nymeria pose root not found at {nymeria_pose_root}")
        return []

    # Iterate over splits
    for split_file in split_files:
        split_path = os.path.join(cfg.nymeria_split_root, split_file)
        if not os.path.exists(split_path):
            guru.warning(f"Split file missing: {split_path}")
            continue
            
        with open(split_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
            
        guru.info(f"Processing split {split_file}: {len(sequences)} sequences")
        
        for seq_name in sequences:
            # Each sequence is a directory containing .npy files
            seq_pose_dir = os.path.join(nymeria_pose_root, seq_name)
            seq_smpl_dir = os.path.join(nymeria_smpl_root, seq_name)
            
            if not os.path.exists(seq_pose_dir):
                # guru.warning(f"Sequence dir not found: {seq_pose_dir}")
                continue
                
            # Get .npy files inside the sequence directory
            # Use same glob/sort logic as get_dataset_files but specific to this dir
            # Note: get_dataset_files uses recursive glob, here we just want files in this dir (or recursive if structure is deeper)
            # Assuming flat .npy files inside sequence dir per user description logic
            
            joints_paths = sorted(glob.glob(os.path.join(seq_pose_dir, '*.npy')))
            smpl_paths = sorted(glob.glob(os.path.join(seq_smpl_dir, '*.npy')))
            
            if not joints_paths:
                 # Check recursive just in case
                 joints_paths = sorted(glob.glob(os.path.join(seq_pose_dir, '**/*.npy'), recursive=True))
                 smpl_paths = sorted(glob.glob(os.path.join(seq_smpl_dir, '**/*.npy'), recursive=True))

            if len(joints_paths) != len(smpl_paths):
                 guru.warning(f"Mismatch in {seq_name}: {len(joints_paths)} vs {len(smpl_paths)}")
            
            file_pairs = list(zip(joints_paths, smpl_paths))
            
            for joints_path, smpl_path in file_pairs:
                try:
                    # Construct output path
                    # We want to preserve structure or standard naming
                    # E.g. out_root/neutral/Nymeria/seq_name/file.npz
                    
                    # Rel path from pose_data_fps_30
                    rel_path = os.path.relpath(joints_path, os.path.join(cfg.data_root, 'pose_data_fps_30'))
                    name, _ = os.path.splitext(rel_path)
                    
                    tasks.append({
                        "joints_path": joints_path,
                        "smpl_path": smpl_path,
                        "out_path": f"{cfg.out_root}/neutral/{name}.npz",
                        "r_out_path": f"{cfg.out_root}/neutral/{name}_reflect.npz",
                        "path_for_skip": rel_path
                    })
                except Exception as e:
                    guru.error(f"Error preparing task for {joints_path}: {e}")
                    
    return tasks

def main(cfg: Config):
    tasks = []
    
    if cfg.process_nymeria:
        guru.info("Processing Nymeria dataset...")
        tasks.extend(get_nymeria_tasks(cfg))
    else:
        # Default AMASS processing
        dsets = AMASS_SPLITS["all"]
        for dset in dsets:
            # Check if dataset folder exists
            dset_path = os.path.join(cfg.data_root, 'pose_data_fps_30', dset)
            if not os.path.exists(dset_path):
                # guru.info(f"Dataset {dset} not found in {dset_path}, skipping.")
                continue
                
            file_pairs = get_dataset_files(cfg.data_root, dset)
            for joints_path, smpl_path in file_pairs:
                # Construct output path based on relative path from pose_data_fps_30
                # E.g. data_root/pose_data_fps_30/CMU/01_01_poses.npy -> CMU/01_01_poses.npy
                try:
                    rel_path = os.path.relpath(joints_path, os.path.join(cfg.data_root, 'pose_data_fps_30'))
                    # Output to neutral/name.npz and neutral/name_reflect.npz
                    # strip extension
                    name, _ = os.path.splitext(rel_path)
                    
                    tasks.append({
                        "joints_path": joints_path,
                        "smpl_path": smpl_path,
                        "out_path": f"{cfg.out_root}/neutral/{name}.npz",
                        "r_out_path": f"{cfg.out_root}/neutral/{name}_reflect.npz",
                        "path_for_skip": rel_path
                    })
                except Exception as e:
                    guru.error(f"Error preparing paths for {joints_path}: {e}")

    dev_ids = cfg.devices
    guru.info(f"devices {dev_ids}")
    guru.info(f"Total tasks: {len(tasks)}")

    if len(dev_ids) <= 1:
        guru.info("processing in sequence")
        for i, task in tqdm(enumerate(tasks)):
            if check_skip(task["path_for_skip"]):
                guru.info(f"skipping {task['path_for_skip']}")
                continue
                
            process_seq(
                task["joints_path"],
                task["smpl_path"],
                task["out_path"],
                cfg.smplh_root,
                dev_ids[i % len(dev_ids)],
                beta_neutral=True,
                reflect=False,
                overwrite=cfg.overwrite,
            )
            process_seq(
                task["joints_path"],
                task["smpl_path"],
                task["r_out_path"],
                cfg.smplh_root,
                dev_ids[i % len(dev_ids)],
                beta_neutral=True,
                reflect=True,
                overwrite=cfg.overwrite,
            )
        return

    with ProcessPoolExecutor(max_workers=len(dev_ids)) as exe:
        for i, task in tqdm(enumerate(tasks)):
            if check_skip(task["path_for_skip"]):
                guru.info(f"skipping {task['path_for_skip']}")
                continue
                
            exe.submit(
                process_seq,
                task["joints_path"],
                task["smpl_path"],
                task["out_path"],
                cfg.smplh_root,
                dev_ids[i % len(dev_ids)],
                beta_neutral=True,
                reflect=False,
                overwrite=cfg.overwrite,
            )
            exe.submit(
                process_seq,
                task["joints_path"],
                task["smpl_path"],
                task["r_out_path"],
                cfg.smplh_root,
                dev_ids[i % len(dev_ids)],
                beta_neutral=True,
                reflect=True,
                overwrite=cfg.overwrite,
            )


if __name__ == "__main__":
    main(tyro.cli(Config))
