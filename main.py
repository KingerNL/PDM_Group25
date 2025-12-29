import numpy as np
import csv
import time
import matplotlib.pyplot as plt

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.bicycle_model import BicycleModel

from source_files.MPPI import MPPIControllerForPathTracking

from source_files.obstacles_spec_generator import add_obstacleArray_to_env, add_visual_marker #Custom script that has several scenarios containing objects to place in the enviroment

from source_files.rrt_dubin_felienc import rrt_main

dt = 0.025
scaling = 0.3
wheel_radius = 0.31265
wheel_base = 0.494
max_steer_abs = 0.7
max_accel_abs = 50.0
samples_per_dt = 15
horizon_step_T = 75
ref_vel = 2




def run_prius_main(replay = False, n_steps=10000, dt=dt):

###-------------------------------------------Creating the enviroment------------------------------------###
    robots = [
        BicycleModel(
            urdf='prius.urdf',
            mode="vel",
            scaling=scaling,
            wheel_radius=wheel_radius,
            wheel_distance=wheel_base,
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
    ob, _ = env.reset()

    testArray = np.array(([                 #Test array (x, y, radius)
                    [8.0, 8.0, 1.5]]))
    
    _ , all_vertices = add_obstacleArray_to_env(env, testArray)

###---------------------------------------------RRT with dublins path-------------------------------------###
    best_path = rrt_main(all_vertices)
    ref_path = np.array(best_path)

    step = 2
    downsampled = ref_path[::step]

    # Ensure first and last points are included
    if not np.array_equal(downsampled[0], ref_path[0]):
        downsampled = np.vstack([ref_path[0], downsampled])
    if not np.array_equal(downsampled[-1], ref_path[-1]):
        downsampled = np.vstack([downsampled, ref_path[-1]])

    # Add markers
    for i, (x, y, yaw) in enumerate(downsampled):
        if i == 0:
            color = (0.0, 1.0, 0.0, 1.0)  # first point = green
        elif i == len(downsampled) - 1:
            color = (1.0, 0.0, 0.0, 1.0)  # last point = red
        else:
            color = (0.0, 0.0, 1.0, 0.2)  # all others = blue transparent

        add_visual_marker([x, y, 0.02], rgba=color)

### --------------------------------------------------MPPI-------------------------------------------------###
    # load and visualize reference path
    vel_column = np.full((ref_path.shape[0], 1), ref_vel)  # shape (10, 1) filled with 5
    ref_path = np.hstack((ref_path, vel_column))

    x = ref_path[:, 0]  # first column
    y = ref_path[:, 1]  # second column

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 15)

    # Plot path
    ax.plot(x, y, marker='o', linestyle='-', color='b', label='Path')

    # Add black circles
    for i in range(len(testArray)):
        circle = plt.Circle((testArray[i, 0], testArray[i, 1]), testArray[i, 2], color='black', fill=True)
        ax.add_patch(circle)

    # Labels, grid, and aspect ratio
    ax.set_title("Path with Black Dots")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    # Step 4: Save the plot to a file instead of showing it
    plt.savefig("Data/path_plot.png", dpi=300)  # saves as PNG with 300 dpi
    print("Plot saved as 'path_plot.png'")

    #variables
    action = np.zeros(2)  # [velocity, steering_angle]

    mppi = MPPIControllerForPathTracking(
        delta_t = dt, # [s]
        wheel_base = wheel_base * scaling, # [m]
        max_steer_abs = max_steer_abs, # [rad]
        max_accel_abs = max_accel_abs, # [m/s^2]
        ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
        horizon_step_T = horizon_step_T, # [steps]
        number_of_samples_K = samples_per_dt, # [samples]
        param_exploration = 0.05,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 5.0, 30.0]), # weight for [x, y, yaw, v]
        terminal_cost_weight = np.array([50.0, 50.0, 5.0, 30.0]), # weight for [x, y, yaw, v]
        visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
        obstacle_circles = testArray, # [obs_x, obs_y, obs_radius]
        collision_safety_margin_rate = 0.1, # safety margin for collision check
    )

###-----------------------main simulation loop for creating control input or replaying----------------------###
    if (replay == False):
        #delete the old replay data.
        open("Data/MPPI_control_input.csv" , "w").close()
        
        for _ in range(n_steps):
            #get the current state from the env
            pos = ob['robot_0']['joint_state']['position']
            x, y, yaw = pos
            forward_vel, side_vel = ob['robot_0']['joint_state']['forward_velocity']
            # print(ob['robot_0']['joint_state']['forward_velocity'])
            steering = ob['robot_0']['joint_state']['steering'][0]

            # State vector for MPPI
            current_state = np.array([x, y, yaw, forward_vel])
            
            # calculate input force with MPPI
            try:
                optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                    observed_x = current_state
                )
            except IndexError as e:
                print("[ERROR] IndexError detected. Terminate simulation. The vehicle has reached the end of the referenc path")
                break

            #create state for the env
            current_vel = forward_vel
            next_vel = current_vel + optimal_input[1] * dt

            action[0] = next_vel
            action[1] = optimal_input[0]
            ob, *_ = env.step(action)

            print(optimal_traj)

            #Put the state that is calcultated for the mppi into a csv file
            with open("Data/MPPI_control_input.csv" , "a" , newline="") as f:
                writer = csv.writer(f)
                writer.writerow(action)

            #print the state and send to the env
            # print(action)
            ob, *_ = env.step(action)
    
    else:
        #if replay is true, it will load the file and play it in the env.
        loaded = np.loadtxt("Data/MPPI_control_input_succes1.csv" , delimiter=",")
        for i in range(loaded.shape[0]):
            ob, *_ = env.step(loaded[i])


    #sleep before closing the env
    time.sleep(5)
    print("close")
    env.close()


###----------------------------------------------------MAIN-----------------------------------------------------------###
if __name__ == "__main__":
    run_prius_main(replay=False)
