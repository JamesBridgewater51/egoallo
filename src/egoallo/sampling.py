from __future__ import annotations

import time

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from . import fncsmpl, network
from .guidance_optimizer_jax import GuidanceMode, do_guidance_optimization
from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from .tensor_dataclass import TensorDataclass
from .transforms import SE3


def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]


class CosineNoiseScheduleConstants(TensorDataclass):
    alpha_t: Float[Tensor, "T"]
    r"""$1 - \beta_t$"""

    alpha_bar_t: Float[Tensor, "T+1"]
    r"""$\Prod_{j=1}^t (1 - \beta_j)$"""

    @staticmethod
    def compute(timesteps: int, s: float = 0.008) -> CosineNoiseScheduleConstants:
        steps = timesteps + 1
        x = torch.linspace(0, 1, steps, dtype=torch.float64)

        def get_betas():
            alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        alpha_t = 1.0 - get_betas()
        assert len(alpha_t.shape) == 1
        alpha_cumprod_t = torch.cat(
            [torch.ones((1,)), torch.cumprod(alpha_t, dim=0)],
            dim=0,
        )
        return CosineNoiseScheduleConstants(
            alpha_t=alpha_t, alpha_bar_t=alpha_cumprod_t
        )


def run_sampling_with_stitching(
    denoiser_network: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    guidance_post: bool,
    guidance_inner: bool,
    Ts_world_cpf: Float[Tensor, "time 7"],
    floor_z: float,
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    num_samples: int,
    device: torch.device,
    guidance_verbose: bool = True,
) -> network.EgoDenoiseTraj:
    # Offset the T_world_cpf transform to place the floor at z=0 for the
    # denoiser network. All of the network outputs are local, so we don't need to
    # unoffset when returning.
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz

    x_t_packed = torch.randn(
        (num_samples, Ts_world_cpf.shape[0] - 1, denoiser_network.get_d_state()),
        device=device,
    )
    x_t_list = [
        network.EgoDenoiseTraj.unpack(
            x_t_packed, include_hands=denoiser_network.config.include_hands
        )
    ]
    ts = quadratic_ts()

    seq_len = x_t_packed.shape[1]

    start_time = None

    window_size = 128
    overlap_size = 32
    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                # Make this shape /```\
                overlap_size,
                np.minimum(
                    # Make this shape: /
                    np.arange(1, seq_len + 1),
                    # Make this shape: \
                    np.arange(1, seq_len + 1)[::-1],
                ),
            )
            / overlap_size,
        )
        .to(device)
        .to(torch.float32)
    )
    for i in tqdm(range(len(ts) - 1)):
        print(f"Sampling {i}/{len(ts) - 1}")
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            # Chop everything into windows.
            x_0_packed_pred = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=x_t_packed.device)

            # Denoise each window.
            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                assert end_t - start_t > 0
                overlap_weights_slice = canonical_overlap_weights[
                    None, : end_t - start_t, None
                ]
                overlap_weights[:, start_t:end_t, :] += overlap_weights_slice
                x_0_packed_pred[:, start_t:end_t, :] += (
                    denoiser_network.forward(
                        x_t_packed[:, start_t:end_t, :],
                        torch.tensor([t], device=device).expand((num_samples,)),
                        T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(
                            (num_samples, 1, 1)
                        ),
                        T_world_cpf=Ts_world_cpf_shifted[
                            None, start_t + 1 : end_t + 1, :
                        ].repeat((num_samples, 1, 1)),
                        project_output_rotmats=False,
                        hand_positions_wrt_cpf=None,  # TODO: this should be filled in!!
                        mask=None,
                    )
                    * overlap_weights_slice
                )

            # Take the mean for overlapping regions.
            x_0_packed_pred /= overlap_weights

            x_0_packed_pred = network.EgoDenoiseTraj.unpack(
                x_0_packed_pred,
                include_hands=denoiser_network.config.include_hands,
                project_rotmats=True,
            ).pack()

        if torch.any(torch.isnan(x_0_packed_pred)):
            print("found nan", i)
        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        )

        if guidance_mode != "off" and guidance_inner:
            x_0_pred, _ = do_guidance_optimization(
                # It's important that we _don't_ use the shifted transforms here.
                Ts_world_cpf=Ts_world_cpf[1:, :],
                traj=network.EgoDenoiseTraj.unpack(
                    x_0_packed_pred, include_hands=denoiser_network.config.include_hands
                ),
                body_model=body_model,
                guidance_mode=guidance_mode,
                phase="inner",
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
                verbose=guidance_verbose,
            )
            x_0_packed_pred = x_0_pred.pack()
            del x_0_pred

        if start_time is None:
            start_time = time.time()

        # print(sigma_t)
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
        )
        x_t_list.append(
            network.EgoDenoiseTraj.unpack(
                x_t_packed, include_hands=denoiser_network.config.include_hands
            )
        )

    if guidance_mode != "off" and guidance_post:
        constrained_traj = x_t_list[-1]
        constrained_traj, _ = do_guidance_optimization(
            # It's important that we _don't_ use the shifted transforms here.
            Ts_world_cpf=Ts_world_cpf[1:, :],
            traj=constrained_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
            verbose=guidance_verbose,
        )
        assert start_time is not None
        print("RUNTIME (exclude first optimization)", time.time() - start_time)
        return constrained_traj
    else:
        assert start_time is not None
        print("RUNTIME (exclude first optimization)", time.time() - start_time)
        return x_t_list[-1]


def run_batched_sampling(
    denoiser_network: network.EgoDenoiser,
    Ts_world_cpf: Float[Tensor, "batch time 7"],
    num_samples: int,
    device: torch.device,
    floor_z: float = 0.0,
) -> network.EgoDenoiseTraj:
    """
    Run sampling for a batch of sequences.
    Assumes Ts_world_cpf is (Batch, Time, 7).
    No guidance optimization.
    No sliding window (processes full sequence length).
    """
    batch_size = Ts_world_cpf.shape[0]
    timesteps_seq = Ts_world_cpf.shape[1]

    # Offset floor
    Ts_world_cpf_shifted = Ts_world_cpf.clone()
    Ts_world_cpf_shifted[..., 6] -= floor_z

    # Compute relative transforms: (B, T-1, 7)
    # SE3 inverse and matmul support batching
    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz

    # Expand to num_samples: (B * num_samples, T-1, 7)
    # T_cpf_tm1_cpf_t: (B, T-1, 7) -> (B, 1, T-1, 7) -> (B, num_samples, T-1, 7) -> (B*num_samples, T-1, 7)
    T_cpf_tm1_cpf_t_expanded = T_cpf_tm1_cpf_t[:, None, ...].repeat(1, num_samples, 1, 1).reshape(batch_size * num_samples, timesteps_seq - 1, 7)
    
    # Similarly for Ts_world_cpf_shifted: we need it for T_world_cpf arg in forward
    # Ts_world_cpf_shifted: (B, T, 7). Network expects T starting from t+1.
    Ts_world_cpf_shifted_expanded = Ts_world_cpf_shifted[:, None, ...].repeat(1, num_samples, 1, 1).reshape(batch_size * num_samples, timesteps_seq, 7)


    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t

    # Initialize noise: (B * num_samples, T-1, d_state)
    d_state = denoiser_network.get_d_state()
    x_t_packed = torch.randn(
        (batch_size * num_samples, timesteps_seq - 1, d_state),
        device=device,
    )
    
    ts = quadratic_ts()

    for i in tqdm(range(len(ts) - 1), desc="Batched Sampling"):
        t = ts[i]
        t_next = ts[i + 1]

        with torch.inference_mode():
            # forward expects:
            # x_t: (N, T, D)
            # t: (N,)
            # T_cpf_tm1_cpf_t: (N, T, 7)
            # T_world_cpf: (N, T, 7) (corresponding to t+1 to end)

            x_0_packed_pred = denoiser_network.forward(
                x_t_packed,
                torch.tensor([t], device=device).expand((batch_size * num_samples,)),
                T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t_expanded,
                T_world_cpf=Ts_world_cpf_shifted_expanded[:, 1:, :], 
                project_output_rotmats=True,
                hand_positions_wrt_cpf=None,
                mask=None,
            )

        # DDIM update
        # if torch.any(torch.isnan(x_0_packed_pred)):
        #     print("found nan", i)
            
        sigma_t = torch.cat(
            [
                torch.zeros((1,), device=device),
                torch.sqrt(
                    (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                )
                * 0.8,
            ]
        )

        
        x_t_packed = (
            torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
            + (
                torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
            )
            + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
        )

    # Finally unpack
    # network.EgoDenoiseTraj.unpack expects (N, T, D).
    flat_traj = network.EgoDenoiseTraj.unpack(
        x_t_packed,
        include_hands=denoiser_network.config.include_hands,
        project_rotmats=True
    )
    
    return flat_traj
