"""Example script for computing body metrics on the test split of the AMASS/Nymeria dataset with batching.
"""

from pathlib import Path
from typing import List, TypeVar


import numpy as np
import torch
import torch.utils.data
import tyro
from tqdm.auto import tqdm

from egoallo import fncsmpl, network
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
from egoallo.inference_utils import load_denoiser
from egoallo.metrics_helpers import (
    compute_foot_contact,
    compute_foot_skate,
    compute_head_trans,
    compute_mpjpe,
)
from egoallo.sampling import run_batched_sampling
from egoallo.transforms import SE3, SO3
from egoallo.tensor_dataclass import TensorDataclass

T = TypeVar("T")

def _stack_tensor_dataclass(items: List[T]) -> T:
    """Stack a list of TensorDataclasses."""
    if not items:
        return None
    
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    elif isinstance(first, TensorDataclass):
        # Recursively stack fields
        fields = vars(first).keys()
        new_data = {}
        for f in fields:
            new_data[f] = _stack_tensor_dataclass([getattr(item, f) for item in items])
        return type(first)(**new_data)
    elif isinstance(first, (list, tuple)):
         # Assuming fixed length lists/tuples for all items
        return type(first)(_stack_tensor_dataclass([item[i] for item in items]) for i in range(len(first)))
    elif isinstance(first, dict):
        return {k: _stack_tensor_dataclass([item[k] for item in items]) for k in first}
    elif first is None:
        return None
    else:
        # Fallback for non-tensor fields (e.g. metadata strings), just return list
        return items

def collate_tensor_dataclass(batch: List[T]) -> T:
    return _stack_tensor_dataclass(batch)


def main(
    dataset_hdf5_path: Path,
    dataset_files_path: Path,
    subseq_len: int = 128,
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/"),
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    num_samples: int = 1,
    batch_size: int = 1,
) -> None:
    """Compute body metrics on the test split with batching."""
    device = torch.device("cuda")

    # Setup.
    denoiser_network = load_denoiser(checkpoint_dir).to(device)
    
    # Dataset
    # slice_strategy="deterministic" ensures consistent evaluation
    dataset = EgoAmassHdf5Dataset(
        dataset_hdf5_path,
        dataset_files_path,
        splits=("test",),
        subseq_len=subseq_len + 1,
        cache_files=True,
        slice_strategy="deterministic",
        random_variable_len_proportion=0.0,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, # Simplify for now
        collate_fn=collate_tensor_dataclass
    )

    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)
    metrics = list[dict[str, np.ndarray]]()

    print(f"Starting evaluation with batch_size={batch_size}, subseq_len={subseq_len}")
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Move batch to device
        batch_data = batch_data.to(device)
        current_batch_size = batch_data.T_world_cpf.shape[0]

        # Run Batched Sampling
        # Output: flat_traj (TensorDataclass) with shape (B*num_samples, T, ...)
        # Note: run_batched_sampling returns FLAT trajectory
        flat_samples = run_batched_sampling(
            denoiser_network,
            Ts_world_cpf=batch_data.T_world_cpf,
            num_samples=num_samples,
            device=device,
            floor_z=0.0,
        )

        # Reshape stats for iteration
        # Need to reshape fields from (B*N, ...) to (B, N, ...) to process per sample
        def reshape_to_batch_sample(t: torch.Tensor):
            if isinstance(t, torch.Tensor) and t.shape[0] == current_batch_size * num_samples:
                return t.view(current_batch_size, num_samples, *t.shape[1:])
            return t

        batched_samples = flat_samples.map(reshape_to_batch_sample)
        
        # Iterate over batch elements to compute metrics
        for b in range(current_batch_size):
            # Extract single sample (contains N hypotheses)
            # We can select from batched_samples
            def select_b(t: torch.Tensor):
                return t[b]
            
            samples_b = batched_samples.map(select_b) # (num_samples, T, ...)
            
            # Extract ground truth for b
            sequence_b = batch_data.map(select_b) # (T, ...)

            # Prepare predictions (Forward Kinematics)
            # samples_b.body_rotmats: (N, T, 21, 3, 3)
            # samples_b.betas: (N, T, 16)
            
            # Reshape betas if needed (model expects one shape per batch usually, or broadcast)
            # body_model.with_shape expects (..., 16)
            
            pred_posed = body_model.with_shape(samples_b.betas).with_pose(
                T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
                local_quats=SO3.from_matrix(
                    torch.cat([samples_b.body_rotmats, samples_b.hand_rotmats], dim=2)
                ).wxyz,
            )
            
            # Post-process T_world_root
            # sequence_b.T_world_cpf[1:] corresponds to our T output frames
            # samples_b output T frames. sequence_b has subseq_len+1 frames.
            # We need to match. run_batched_sampling outputs T-1 frames (relative to input T).
            # Wait, run_batched_sampling:
            # Inputs: Ts_world_cpf (B, T_in, 7)
            # Outputs: x_t_packed (B*N, T_in-1, ...)
            # So output length is T_in - 1.
            
            pred_posed = pred_posed.with_new_T_world_root(
                get_T_world_root_from_cpf_pose(pred_posed, sequence_b.T_world_cpf[1:, ...])
            )

            # Ground Truth Posed
            # sequence_b: subseq_len+1. We compare 1:end.
            label_posed = body_model.with_shape(sequence_b.betas[1:, ...]).with_pose(
                sequence_b.T_world_root[1:, ...],
                torch.cat(
                    [
                        sequence_b.body_quats[1:, ...],
                        sequence_b.hand_quats[1:, ...],
                    ],
                    dim=1,
                ),
            )
            
            # Compute Metrics
            metrics.append(
                {
                    "mpjpe": compute_mpjpe(
                        label_T_world_root=label_posed.T_world_root,
                        label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                        pred_T_world_root=pred_posed.T_world_root,
                        pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                        per_frame_procrustes_align=False,
                    ),
                    "pampjpe": compute_mpjpe(
                        label_T_world_root=label_posed.T_world_root,
                        label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                        pred_T_world_root=pred_posed.T_world_root,
                        pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                        per_frame_procrustes_align=True,
                    ),
                    "foot_skate": compute_foot_skate(
                        pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    ),
                    "foot_contact (GND)": compute_foot_contact(
                        pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    ),
                    "T_head": compute_head_trans(
                        label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                        pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    ),
                }
            )

    print("=" * 80)
    print("Final Metrics")
    metric_keys = metrics[0].keys() if metrics else []
    
    if not metrics:
        print("\tNo metrics collected.")
    else:
        for k in metric_keys:
            # Gather all values for this metric key across the batch
            values = [m[k] for m in metrics]
            # Compute stats
            mean_val = np.mean(values)
            std_val = np.std(values)
            sem_val = std_val / np.sqrt(len(metrics) * num_samples)
            
            print(f"\t {k} {mean_val:.3f} +/- {sem_val:.3f}")

    print("=" * 80)

if __name__ == "__main__":
    tyro.cli(main)
