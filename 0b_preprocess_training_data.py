"""Translate data from HuMoR-style npz format to an hdf5-based one.

Due to AMASS licensing, we unfortunately can't re-distribute our preprocessed dataset. If you have questions
or run into issues, please reach out.
"""

import queue
import threading
import time
from pathlib import Path

import h5py
import torch
import torch.cuda
import tyro

from egoallo import fncsmpl, vis_helpers
from egoallo.data.amass import EgoTrainingData

import viser
import viser.transforms as vtf


def main(
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    data_npz_dir: Path = Path("./data/processed_30fps_no_skating/"),
    output_file: Path = Path("./data/egoalgo_no_skating_dataset.hdf5"),
    output_list_file: Path = Path("./data/egoalgo_no_skating_dataset_files.txt"),
    include_hands: bool = True,
    vis: bool = False,
) -> None:
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path)

    assert torch.cuda.is_available()

    task_queue = queue.Queue[Path]()
    for path in list(data_npz_dir.glob("**/*.npz")):
        task_queue.put_nowait(path)

    total_count = task_queue.qsize()
    start_time = time.time()

    output_hdf5 = h5py.File(output_file, "w")
    file_list: list[str] = []

    def worker(device_idx: int) -> None:
        device_body_model = body_model.to("cuda:" + str(device_idx))

        while True:
            try:
                npz_path = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f"Processing {npz_path} on device {device_idx}...")
            train_data = EgoTrainingData.load_from_npz(
                device_body_model, npz_path, include_hands=include_hands
            )

            assert "neutral" in str(npz_path)
            group_name = str(npz_path).rpartition("neutral/")[2]

            if vis:
                print(f"Visualizing {group_name}...")
                server = viser.ViserServer(port=8080)
                
                # Mock trajectory object for visualization
                class MockTraj:
                    pass
                
                traj = MockTraj()
                # EgoTrainingData has betas as (1, 10) or (1, 16). 
                # Need to tile to (1, T, 16)
                timesteps = train_data.T_world_cpf.shape[0]
                
                # Ensure betas is 16 dim
                betas = train_data.betas # (1, 10 or 16)
                if betas.shape[1] == 10:
                    betas = torch.cat([betas, torch.zeros((1, 6)).to(betas.device)], dim=1)
                
                traj.betas = betas.unsqueeze(0).repeat(1, timesteps, 1) # (1, T, 16)
                
                # Quats to Rotmats
                # EgoTrainingData body_quats: (T, 21, 4) -> (time, joint, wxyz)
                # Helper expects (sample, time, joint, 3, 3)
                traj.body_rotmats = vtf.SO3(train_data.body_quats.numpy(force=True)).as_matrix()
                traj.body_rotmats = torch.from_numpy(traj.body_rotmats).unsqueeze(0)
                
                if train_data.hand_quats is not None:
                    # (T, 30, 4)
                    traj.hand_rotmats = vtf.SO3(train_data.hand_quats.numpy(force=True)).as_matrix()
                    traj.hand_rotmats = torch.from_numpy(traj.hand_rotmats).unsqueeze(0)
                else:
                    traj.hand_rotmats = None
                    
                # Contacts for coloring. Use dummy or data if available.
                # EgoTrainingData contacts: (T, 4) - feet contacts?
                # Helper expects contacts for all 21 joints if show_joints=True?
                # Actually helper uses: traj.contacts[j, t, :] which is (21,) colors?
                # Let's inspect helper again:
                # joints_colors[:, 0] = traj.contacts[j, t, :].numpy()
                # It expects contacts to be (sample, time, 21) or similar? 
                # Let's just pass zeros to avoid crash if we don't have full contact info
                traj.contacts = torch.zeros((1, timesteps, 21))
                
                update_cb = vis_helpers.visualize_traj_and_hand_detections(
                    server=server,
                    Ts_world_cpf=train_data.T_world_cpf,
                    traj=traj,
                    body_model=device_body_model.to("cpu"), # vis helper often runs on CPU/numpy
                    show_joints=True
                )
                
                print("Visualization server running at http://localhost:8080. Press Ctrl+C to stop or close.")
                try:
                    # while True:
                    for _ in range(len(train_data.T_world_cpf)):
                        update_cb()
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    print("Stopping visualization and continuing...")
                    pass

            print(f"Writing to group {group_name} on {device_idx}...")
            group = output_hdf5.create_group(group_name)
            file_list.append(group_name)

            for k, v in vars(train_data).items():
                # No need to write the mask, which will always be ones when we
                # load from the npz file!
                if k == "mask":
                    continue

                # Chunk into 32 timesteps at a time.
                assert v.dtype == torch.float32
                if v.shape[0] == train_data.T_world_cpf.shape[0]:
                    chunks = (min(32, v.shape[0]),) + v.shape[1:]
                else:
                    assert v.shape[0] == 1
                    chunks = v.shape
                group.create_dataset(k, data=v.numpy(force=True), chunks=chunks)

            print(
                f"Finished ~{total_count - task_queue.qsize()}/{total_count},",
                f"{(total_count - task_queue.qsize()) / total_count * 100:.2f}% in",
                f"{time.time() - start_time} seconds",
            )

    workers = [
        threading.Thread(target=worker, args=(i,))
        for i in range(torch.cuda.device_count())
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    output_list_file.write_text("\n".join(file_list))


if __name__ == "__main__":
    tyro.cli(main)
