import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.bicycle_model import BicycleModel

from keyboard_control import KeyboardController

import create_enviroment

import obstacles_spec_generator

def run_prius_keyboard(n_steps=10000, dt=0.01):

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

    obstacles_specs = obstacles_spec_generator.static_scenario_1()

    create_enviroment.place_obstacles_in_env(env, obstacles_specs)

    controller = KeyboardController()
    controller.start()

    # main simulation loop
    for _ in range(n_steps):
        action = controller.step(dt)   # uses old state + input - drag
        env.step(action)

    env.close()


if __name__ == "__main__":
    run_prius_keyboard()

