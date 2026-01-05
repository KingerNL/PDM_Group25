import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.bicycle_model import BicycleModel

from keyboard_control import KeyboardController

import create_enviroment.create_enviroment as create_enviroment        #Custom script that places functions inside the enviroment

def run_prius_keyboard(n_steps=10000, dt=0.01):

    robots = [
        BicycleModel(
            urdf='prius.urdf',
            mode="vel",
            scaling=0.3,
            wheel_radius=0.31265,
            wheel_distance=0.494,
            spawn_offset=np.array([0, 0, 0.05]),        #Can also be any value for x and y between -15 and +15
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

    start_x = ob[0]['robot_0']['joint_state']['position'][0]    #Get current/start position
    start_y = ob[0]['robot_0']['joint_state']['position'][1]    #Get current/start position
    start_position = [start_x, start_y, 0.05]
    create_enviroment.add_visual_marker(start_position, rgba=(1.0, 0.0, 0.0, 0.6))   #Create non-colliding marker at start
    
    #testArray = np.array(([                 #Test array of obstacles to be placed (x, y, radius)
    #                [2.0, 2.0, 1.0],
    #                [-4.0, -4.0, 1.0]]))
    #create_enviroment.add_obstacleArray_to_env(env, testArray)

    obstacle_array = create_enviroment.generate_random_obstacle_array(num_points=50, min_dist=1.5, max_radius=3.0, robot_pos=start_position[:2])
    create_enviroment.add_obstacleArray_to_env(env, obstacle_array)

    goal_position = [1.0, 1.0, 0.05]
    create_enviroment.add_visual_marker(goal_position)   #Create non-colliding marker at goal

    controller = KeyboardController()
    controller.start()

    # main simulation loop
    for _ in range(n_steps):
        action = controller.step(dt)   # uses old state + input - drag
        env.step(action)

    env.close()


if __name__ == "__main__":
    run_prius_keyboard()