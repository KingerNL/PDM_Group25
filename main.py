import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.bicycle_model import BicycleModel

from MPPI import MPPIControllerForPathTracking


def run_prius_main(n_steps=10000, dt=0.01):

    robots = [
        BicycleModel(
            urdf='prius.urdf',
            mode="vel",
            scaling=0.3,
            wheel_radius=0.31265,
            wheel_distance=0.494,
            actuated_wheels=[
                'front_right_wheel_joint', 'front_left_wheel_joint',
                'rear_right_wheel_joint', 'rear_left_wheel_joint'
            ],
            steering_links=[
                'front_right_steer_joint', 'front_left_steer_joint'
            ],
            facing_direction='-x'
        )
    ]

    env = UrdfEnv(dt=dt, robots=robots, render=True)
    ob = env.reset()
    print("Initial observation:", ob)
    


### --------------------------------------------------MPPI-------------------------------------------------###
    # load and visualize reference path
    ref_path = np.genfromtxt('Data/ovalpath.csv', delimiter=',', skip_header=1)


    ### CSV to np.array!!!!!!!!!!!!!!!
    OBSTACLE_CIRCLES = np.array([
        [+ 8.0, +5.0, 4.0], # pos_x, pos_y, radius [m] in the global frame
        [+18.0, -5.0, 4.0], # pos_x, pos_y, radius [m] in the global frame
    ])



    mppi = MPPIControllerForPathTracking(
        delta_t = dt, # [s]
        wheel_base = 2.5, # [m]
        max_steer_abs = 0.523, # [rad]
        max_accel_abs = 2.000, # [m/s^2]
        ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
        horizon_step_T = 20, # [steps]
        number_of_samples_K = 500, # [samples]
        param_exploration = 0.05,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        terminal_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        visualze_sampled_trajs = True, # if True, sampled trajectories are visualized
        obstacle_circles = OBSTACLE_CIRCLES, # [obs_x, obs_y, obs_radius]
        collision_safety_margin_rate = 1.2, # safety margin for collision check
    )

###-------------------------------------------main simulation loop---------------------------------------------------###
    for _ in range(n_steps):
        robot_dict = ob[0]

        # unpack position: [x, y, yaw]
        x, y, yaw = robot_dict['robot_0']['joint_state']['position']

        # unpack forward & side velocity
        forward_vel, side_vel = robot_dict['robot_0']['joint_state']['forward_velocity']

        # steering is an array with one element
        steering = robot_dict['robot_0']['joint_state']['steering'][0]

        # build state vector (using yaw from position or using steering — depends on your model)
        current_state = np.array([x, y, yaw, forward_vel])


        try:
            # calculate input force with MPPI
            optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                observed_x = current_state
            )
            print(optimal_input)
        except IndexError as e:
            # the vehicle has reached the end of the reference path
            print("[ERROR] IndexError detected. Terminate simulation.")
            break

        action = np.array([2.,1.])
        
        ### FIX that the env gets the right input
        ob, *_ = env.step(action)
        # print(ob)

    env.close()


###----------------------------------------------------MAIN-----------------------------------------------------------###

if __name__ == "__main__":
    run_prius_main()

