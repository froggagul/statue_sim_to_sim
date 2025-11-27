import argparse
import os
import time

import torch
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt


class Simulator:
    def __init__(
        self,
        xml_path: str,
        joint_names: list[str],
        env_dt: float = 0.02,
        decimation: int = 4,
        gravity: tuple[float, float, float] = (0, 0, -9.81),
    ):

        # Load model from MJCF XML file.
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize the simulation data.
        mujoco.mj_resetData(self.model, self.data)

        self.pelvis_link_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
        )

        # Set the time step for the simulation.

        self.env_dt = env_dt
        self.decimation = decimation

        self.model.opt.timestep = env_dt / decimation
        # Set the gravity for the simulation.
        self.model.opt.gravity = gravity

        # Placeholder for the viewer
        self.viewer = None

        # Joint names
        self.joint_names = joint_names

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.joint_names
            ]
        )

        self.last_action: npt.NDArray[np.float64] = np.zeros(
            (len(self.actuator_ids),), dtype=np.float64
        )

        self.phase: float = 0.0
        self.target_velocity: npt.NDArray[np.float64] = np.zeros((3,))

        self.joint_offsets: npt.NDArray[np.float64] = np.array(
            [
                -0.3000,
                0.3000,
                0.0000,
                0.0000,
                -0.2000,
                0.2000,
                -0.5000,
                0.5000,
                -0.3000,
                -0.3000,
                0.0000,
                0.0000,
            ]
        )

    # Define a key callback function to handle keyboard inputs.
    def key_callback(self, keycode):
        if keycode == 265:  # up arrow
            self.target_velocity[0] += 0.1
            self.target_velocity = self.target_velocity.clip(-1.0, 1.0)
            print(f"target velocity: {list(self.target_velocity.round(1))}")
        elif keycode == 264:  # down arrow
            self.target_velocity[0] -= 0.1
            self.target_velocity = self.target_velocity.clip(-1.0, 1.0)
            print(f"target velocity: {list(self.target_velocity.round(1))}")
        elif keycode == 263:  # left arrow
            self.target_velocity[1] += 0.1
            self.target_velocity = self.target_velocity.clip(-1.0, 1.0)
            print(f"target velocity: {list(self.target_velocity.round(1))}")
        elif keycode == 262:  # right arrow
            self.target_velocity[1] -= 0.1
            self.target_velocity = self.target_velocity.clip(-1.0, 1.0)
            print(f"target velocity: {list(self.target_velocity.round(1))}")
        else:
            pass
            # print(f"Key pressed: {keycode}")

    def run(self):

        # Launch the viewer if not already launched.
        if self.viewer is not None:
            self.viewer.close()

        # Launch the viewer with the model.
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        )

        # synchronize the model with the viewer
        self.viewer.sync()

    def _pre_process_action(self, action: np.ndarray):
        self.last_action[:] = action

        self.applied_action = self.joint_offsets + action * 0.5
        self.data.ctrl[self.actuator_ids] = self.applied_action

    def _get_observations(self) -> np.ndarray:

        obs_list: list[np.ndarray] = []

        imu_gyro = self.data.sensordata[:3]
        obs_list.append(imu_gyro)

        projected_gravity = project_gravity_calc(self.data.xquat[self.pelvis_link_id])
        obs_list.append(projected_gravity)

        obs_list.append(self.target_velocity)

        if np.linalg.norm(self.target_velocity) > 1e-3:
            self.phase += self.env_dt * np.pi / 0.8 * 2.0
        else:
            self.phase = 0.0

        obs_list.append(np.array([np.sin(self.phase), np.cos(self.phase)]))

        joint_pos = self.data.actuator_length[self.actuator_ids]
        joint_pos -= self.joint_offsets
        obs_list.append(joint_pos)

        joint_vel = self.data.actuator_velocity[self.actuator_ids]
        obs_list.append(joint_vel)

        obs_list.append(self.last_action)

        obs = np.concatenate(obs_list, axis=0).astype(np.float32)

        return obs

    def apply_action(self, action: torch.Tensor):
        """Apply the action to the simulation without stepping."""
        self._pre_process_action(action.cpu().numpy())

    def step(self, latency_step: int = 0):
        """Step the simulation without applying a new action."""

        for _ in range(self.decimation + latency_step):
            mujoco.mj_step(self.model, self.data)

        self.viewer.sync()

    def get_observations(self) -> torch.Tensor:
        """Get the current observations from the simulation."""
        obs = self._get_observations()
        return torch.from_numpy(obs).float()

    def shutdown(self):
        """Shutdown the simulator and close the viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        print("Simulator shut down successfully.")


class MathUtils:

    @staticmethod
    def quat_rotate_inverse(q: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

        Args:
            q: The quaternion in (w, x, y, z). Shape is (..., 4).
            v: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * np.expand_dims((2.0 * q_w**2 - 1.0), axis=-1)
        b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
        # for two-dimensional tensors, bmm is faster than einsum
        if len(q_vec.shape) == 2:
            c = q_vec * (q_vec @ v.T) * 2.0
        else:
            c = q_vec * np.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        return a - b + c


def project_gravity_calc(quat: npt.NDArray) -> npt.NDArray:
    gravity_dir = np.array(
        [
            [0.0, 0.0, -1.0],
        ]
    )

    projected_gravity = MathUtils.quat_rotate_inverse(
        np.expand_dims(quat, axis=0), gravity_dir
    ).squeeze()

    return projected_gravity


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_latency_step",
        type=int,
        default=8,
        help="Number of MuJoCo substeps per control step.",
    )
    args = parser.parse_args()



    root_dir = os.path.dirname(os.path.abspath(__file__))

    xml_path = os.path.join(root_dir, "assets/scene_statue_lowerbody_with10kg.xml")
    policy_path = os.path.join(root_dir, "models/policy.pt")

    joint_names = [
        "left_hip_pitch_m",
        "right_hip_pitch_m",
        "left_hip_roll_m",
        "right_hip_roll_m",
        "left_hip_yaw_m",
        "right_hip_yaw_m",
        "left_knee_pitch_m",
        "right_knee_pitch_m",
        "left_ankle_pitch_m",
        "right_ankle_pitch_m",
        "left_ankle_roll_m",
        "right_ankle_roll_m",
    ]

    simulator = Simulator(
        xml_path,
        joint_names,
        # decimation=16,
    )
    policy = torch.jit.load(policy_path)

    try:
        simulator.run()
        with torch.no_grad():
            while simulator.viewer.is_running():
                time_start = time.time()

                obs = simulator.get_observations()

                actions: torch.Tensor = policy(obs)

                simulator.apply_action(actions)
                latency_step = np.random.randint(0, args.max_latency_step + 1)
                # print(latency_step)
                simulator.step(latency_step)

                time_end = time.time()

                if time_end - time_start < 0.02:
                    time.sleep(0.02 - (time_end - time_start))
                else:
                    pass
                    # print("step time :", time_end - time_start)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        if simulator.viewer:
            simulator.viewer.close()  # Ensure the viewer is closed on exit.
