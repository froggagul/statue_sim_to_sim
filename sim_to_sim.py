import torch
import os 
from typing import List, Union
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple

import time



class Simulator:
    def __init__(self, 
                 xml_path: str,
                 joint_names: List[str],
                 physics_off: bool = True,
                 env_dt: float = 0.01,
                 decimation: int = 4,
                 gravity: Tuple[float] = (0, 0, -9.81),
                 delay_step = 6,
                 ):
        
        # Load model from MJCF XML file.
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # If physics_off is True, disable physics simulation.
        self.physics = not physics_off

        # Initialize the simulation data.
        mujoco.mj_resetData(self.model, self.data)



        # Set the time step for the simulation.

        self.env_dt = env_dt
        self.decimation = decimation

        self.model.opt.timestep = env_dt/decimation
        # Set the gravity for the simulation.
        self.model.opt.gravity = gravity

        # Placeholder for the viewer
        self.viewer = None

        # Joint names
        self.joint_names = joint_names

        actuator_ids_list = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                     for name in self.joint_names]

        self.actuator_ids = np.array(actuator_ids_list)


        self.last_action = np.zeros((2,12))

        self.phase = 0.0

        self.joint_offsets = np.array([-0.3000,  0.3000,  0.0000,  0.0000, -0.2000,  0.2000, -0.5000,  0.5000, -0.3000, -0.3000,  0.0000,  0.0000])

        self.target_buf = np.zeros((delay_step+1,12))



    # Run the simulation indefinitely until the viewer is closed or keyboard interrupted.
    def run(self):

        # Define a key callback function to handle keyboard inputs.
        def key_callback(keycode):
            if chr(keycode) == 'q':
                print("Quit command received. Closing viewer.")
                self.viewer.close()
            if chr(keycode) == 'r':
                print("Resetting simulation.")
                mujoco.mj_resetData(self.model, self.data)

        # Launch the viewer if not already launched.
        if self.viewer is not None:
            self.viewer.close()

        # Launch the viewer with the model.
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)
        
        # synchronize the model with the viewer
        self.viewer.sync()
    
    def _pre_process_action(self,action:np.ndarray):

        self.applyed_action = self.joint_offsets + action * 0.5

        self.last_action[:-1] = self.last_action[1:]
        self.last_action[-1] = self.applyed_action

    def step_action(self):

        self.target_buf[:-1] = self.target_buf[1:]
        self.target_buf[-1] = self.applyed_action

        self.data.ctrl[self.actuator_ids] = self.target_buf[0]

    def _get_observations(self)->np.ndarray:

        joint_pos = self.data.actuator_length[self.actuator_ids]
        joint_vel = self.data.actuator_velocity[self.actuator_ids]

        joint_pos -= self.joint_offsets


        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        sensordata = self.data.sensordata

        imu_gyro=sensordata[:3]

        projected_gravity = project_gravity_calc(self.data.xquat[id])

        # self.phase += 0.02 * np.pi /0.8 * 2.0

        obs = np.concatenate([
            imu_gyro,
            projected_gravity,
            # np.array([1.0,0.0,0.0]),
            np.array([0.0,0.0,0.0]),
            np.array([np.sin(self.phase),np.cos(self.phase)]),
            joint_pos,
            joint_vel,
            # (self.last_action[-2]*0.6 + self.last_action[-1]*0.4)
            # self.last_action[-2]
            self.last_action[-1]
        ]).astype(np.float32)

        return obs
        
    def update(self, action:np.ndarray):
        """Update the simulation state."""

        self._pre_process_action(action)

        for _ in range(self.decimation):
            self.step_action()
            mujoco.mj_step(self.model, self.data)

        self.viewer.sync()

        obs = self._get_observations()

        return obs

        

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

    root_dir = os.path.dirname(os.path.abspath(__file__))

    xml_path = os.path.join(root_dir, "assets/scene_statue_lowerbody_with10kg.xml")
    policy_path = os.path.join(root_dir, "models/policy.pt")

    joint_names = ["left_hip_pitch_m",
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

    simulator = Simulator(xml_path,joint_names,physics_off= False, env_dt = 0.02,decimation = 4,delay_step=3)
    policy = torch.jit.load(policy_path)

    try:
        simulator.run()
        with torch.no_grad():
            action = np.zeros(12)
            obs = simulator.update(action)
            la = action.copy()
            while True:
                actions : torch.Tensor = policy(torch.tensor(obs))
                time_start = time.time()
                obs =  simulator.update(actions.numpy())
                time_end = time.time()
                if time_end-time_start <0.02:
                    time.sleep(0.02 - (time_end - time_start))
                else:
                    print("step time :",time_end - time_start)
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        if simulator.viewer:
            simulator.viewer.close()  # Ensure the viewer is closed on exit.

        