import numpy as np
import csv
import time
import matplotlib.pyplot as plt

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.bicycle_model import BicycleModel

from source_files.MPPI import MPPIControllerForPathTracking

from source_files.create_enviroment import add_obstacleArray_to_env, add_visual_marker , remove_visual_marker, generate_random_obstacle_array, add_unkown_obstacle_to_array #Custom script that has several scenarios containing objects to place in the enviroment

from source_files.rrt_dubin_felienc import rrt_main

#Scenario Variables
select_scenario = 4             #Select which scenario to run
                                # Scenario 1: Simple, 4 Obstacles
                                # Scenario 2: More complex, 16 Obstacles
                                # Scenario 3: Straight line with 1 unknown obstacle
                                # Scenario 4: 12 obstacles with 2 unknown obstacles

min_dist = 1.5                  #Minimum distance to keep between generated obstacles
max_radius = 3.0                #Maximum radius of obstacles to be generated

def run_prius_main(replay = False, n_steps=10000):
    dt = 0.05
    scaling = 0.3
    wheel_radius = 0.31265
    wheel_base = 0.494
    max_steer_abs = 0.8
    max_accel_abs = 2.5
    samples_per_dt = 20
    horizon_step_T = 25
    ref_vel = 2.5
    offset = -12.5

###-------------------------------------------Creating the enviroment------------------------------------###
    robots = [
        BicycleModel(
            urdf='prius.urdf',
            mode="acc",
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
            facing_direction='-x',
            spawn_offset = [offset, offset, 0.05]
        )
    ]
    
    env = UrdfEnv(dt=dt, robots=robots, render=True)
    ob, _ = env.reset()

    start_x = ob['robot_0']['joint_state']['position'][0]    #Get current/start position
    start_y = ob['robot_0']['joint_state']['position'][1]    #Get current/start position
    start_position = [start_x, start_y, 0.05]
    goal_position = [25+offset, 25+offset, 0]

    #obstacleArray = generate_random_obstacle_array(num_points=1, min_dist=1.5, max_radius=3.0, robot_pos=start_position[:2], goal_position=goal_position[:2])
    
    # Scenario 1: Simple, 4 Obstacles
    if select_scenario == 1:
        obstacleArray = generate_random_obstacle_array(num_points=4, min_dist=min_dist, max_radius=max_radius, robot_pos=start_position[:2], goal_position=goal_position[:2], seed_id=17)
    # Scenario 2: More complex, 16 Obstacles
    elif select_scenario == 2:
        obstacleArray = generate_random_obstacle_array(num_points=16, min_dist=min_dist, max_radius=max_radius, robot_pos=start_position[:2], goal_position=goal_position[:2], seed_id=9)
    # Scenario 3: Straight line with 1 unknown obstacle
    elif select_scenario == 3:
        obstacleArray = generate_random_obstacle_array(num_points=1, min_dist=min_dist, max_radius=max_radius, robot_pos=start_position[:2], goal_position=goal_position[:2], seed_id=2)
    # Scenario 4: 12 obstacles with 2 unknown obstacles
    elif select_scenario == 4:
        obstacleArray = generate_random_obstacle_array(num_points=12, min_dist=min_dist, max_radius=max_radius, robot_pos=start_position[:2], goal_position=goal_position[:2], seed_id=17)
###---------------------------------------------RRT with dublins path-------------------------------------###
    if replay == False:
        _ , all_vertices = add_obstacleArray_to_env(env, obstacleArray, offset)

        # Clear previous CSV
        open("Data/ref_path.csv", "w").close()

        # Generate full path
        start_time = time.perf_counter()
        best_path = rrt_main(all_vertices , 2.2)
        rrt_time = time.perf_counter() - start_time
        print(f"RRT planning time: {rrt_time:.6f} seconds")
        ref_path = np.array(best_path) + offset

        #Create unknown obstacles for scenario 3, after generating RRT path
        if select_scenario == 3:
            obstacleArray = add_unkown_obstacle_to_array(obstacleArray, env, ref_path, offset, unknown_amount=1)
        elif select_scenario == 4:
            obstacleArray = add_unkown_obstacle_to_array(obstacleArray, env, ref_path, offset, unknown_amount=2)
        np.savetxt("Data/obstacles_used.csv", obstacleArray, delimiter=",")

        step = 4  # downsample factor
        downsampled = ref_path[::step]

        # Ensure first and last points are included
        if not np.array_equal(downsampled[0], ref_path[0]):
            downsampled = np.vstack([ref_path[0], downsampled])
        if not np.array_equal(downsampled[-1], ref_path[-1]):
            downsampled = np.vstack([downsampled, ref_path[-1]])

        # Add velocity column to downsampled path
        vel_column = np.full((downsampled.shape[0], 1), ref_vel)  # e.g., ref_vel = 5
        ref_path = np.hstack((downsampled, vel_column))
    
        np.savetxt("Data/ref_path.csv", ref_path, delimiter=",")
    
        x = ref_path[:, 0]  # first column
        y = ref_path[:, 1]  # second column

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        # Plot path
        ax.plot(x, y, marker='o', linestyle='-', color='b', label='Path', zorder=1)
        # Add black circles
        for i in range(len(obstacleArray)):
            circle = plt.Circle((obstacleArray[i, 0], obstacleArray[i, 1]), obstacleArray[i, 2], color='black', fill=True)
            ax.add_patch(circle)

        # Labels, grid, and aspect ratio
        ax.set_title("Path with Black Dots")
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        ax.legend()

        # Step 4: Save the plot to a file instead of showing it
        plt.savefig("Data/path_plot.png", dpi=300)  # saves as PNG with 300 dpi
        print("Plot saved as 'path_plot.png'")
    else:   #Play replay
        ref_path = np.loadtxt("Data/ref_path_succes.csv", delimiter=",")

        obstacleArray = np.loadtxt("Data/obstacles_used.csv", delimiter=",")
        add_obstacleArray_to_env(env, obstacleArray, offset)

    # Extract x, y, yaw for visualization
    x = ref_path[:, 0]
    y = ref_path[:, 1]
    yaw = ref_path[:, 2]

    # Plot markers with colors
    for i, (xi, yi, yiw) in enumerate(zip(x, y, yaw)):
        if i == 0:
            color = (0.0, 1.0, 0.0, 1.0)  # first = green
        elif i == len(x) - 1:
            color = (1.0, 0.0, 0.0, 1.0)  # last = red
        else:
            color = (0.0, 0.0, 1.0, 0.4)  # rest = blue transparent
        add_visual_marker([xi, yi, 0.02], rgba= color)


### --------------------------------------------------MPPI-------------------------------------------------###

    #variables
    action = np.zeros(2)  # [velocity, steering_angle]
    body_ids = []
    loop_times = []
    MPPI_times = []
    walked_path = []

    mppi = MPPIControllerForPathTracking(
        delta_t = dt, # [s]
        wheel_base = wheel_base, # [m]
        max_steer_abs = max_steer_abs, # [rad]
        max_accel_abs = max_accel_abs, # [m/s^2]
        ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
        horizon_step_T = horizon_step_T, # [steps]
        number_of_samples_K = samples_per_dt, # [samples]
        param_exploration = 0.05,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.25, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 5.0, 12.5]), # weight for [x, y, yaw, v]
        terminal_cost_weight = np.array([50.0, 50.0, 5.0, 12.5]), # weight for [x, y, yaw, v]
        visualze_sampled_trajs = True, # if True, sampled trajectories are visualized
        obstacle_circles = obstacleArray, # [obs_x, obs_y, obs_radius]
        collision_safety_margin_rate = 0.4, # safety margin for collision check
    )

###-----------------------main simulation loop for creating control input or replaying----------------------###
    if (replay == False):
        #delete the old replay data.
        open("Data/MPPI_control_input.csv" , "w").close()

        for _ in range(n_steps):
            #get the current state from the env
            start_loop_time = time.perf_counter()
            pos = ob['robot_0']['joint_state']['position']
            x, y, yaw = pos
            forward_vel, side_vel = ob['robot_0']['joint_state']['forward_velocity']
            # print(ob['robot_0']['joint_state']['forward_velocity'])
            steering = ob['robot_0']['joint_state']['steering'][0]


            walked_path.append([x , y])
            add_visual_marker([x ,y , 0.02], radius=0.1, rgba=(1.0, 0, 0, 0.7))

            # State vector for MPPI
            current_state = np.array([x, y, yaw, forward_vel])
            
            # calculate input force with MPPI
            start_MPPI_time = time.perf_counter()
            try:
                optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                    observed_x = current_state
                )
            except IndexError as e:
                print("[ERROR] IndexError detected. Terminate simulation. The vehicle has reached the end of the referenc path")
                break
            MPPI_time = time.perf_counter() - start_MPPI_time
            MPPI_times.append(MPPI_time)

            #create state for the env
            action[0] = optimal_input[1]
            action[1] = optimal_input[0]

            rounded_traj = np.round(optimal_traj, 2)
            rounded_sampled_traj = np.round(sampled_traj_list[:10], 2)
            
            for i in body_ids:
                remove_visual_marker(i)
            body_ids.clear()


            #Show MMPI itterative points
            #for traj in rounded_sampled_traj:
            #    for point in traj[7::3]:
            #        body_id = add_visual_marker([point[0], point[1], 0.02], radius=0.05, rgba=(0.2, 0.2, 0.2, 0.5))
            #        body_ids.append(body_id)

            #Show last MPPI points
            for point in rounded_traj[15::9]:
                body_id = add_visual_marker([point[0], point[1], 0.02], radius=0.1, rgba=(1.0, 0, 0, 0.7))
                body_ids.append(body_id)

            



            #Put the state that is calcultated for the mppi into a csv file
            with open("Data/MPPI_control_input.csv" , "a" , newline="") as f:
                writer = csv.writer(f)
                writer.writerow(action)

            #print the state and send to the env
            print("action acc= ", np.round(action[0],3) , " and steering= " , np.round(action[1],3))
            ob, *_ = env.step(action)
            loop_time = time.perf_counter() - start_loop_time
            loop_times.append(loop_time)
        

        ## create path for saved foto
        walked_path = np.array(walked_path)
        x_walked = walked_path[:, 0]  # first column
        y_walked = walked_path[:, 1]  # second column
        ax.plot(x_walked, y_walked, marker='o', linestyle='-', color='r', label='Path_walked')
        plt.savefig("Data/path_plot_walked.png", dpi=300)  # saves as PNG with 300 dpi
        ax.legend()
        print("Plot saved as 'path_plot_walked.png'")

        print(f"RRT planning time: {rrt_time:.6f} seconds")
        avg_loop_time = sum(loop_times) / len(loop_times)
        print(f"Average loop time: {avg_loop_time:.6f} seconds")
        avg_MPPI_time = sum(MPPI_times) / len(MPPI_times)
        print(f"Average MPPI time: {avg_MPPI_time:.6f} seconds")
    
    else:
        #if replay is true, it will load the file and play it in the env.

        loaded_mppi = np.loadtxt("Data/MPPI_control_input_succes.csv" , delimiter=",")
        for i in range(loaded_mppi.shape[0]):        
            ob, *_ = env.step(loaded_mppi[i])


    #sleep before closing the env
    time.sleep(5)
    print("close")
    env.close()


###----------------------------------------------------MAIN-----------------------------------------------------------###
if __name__ == "__main__":
    run_prius_main(replay=False)
