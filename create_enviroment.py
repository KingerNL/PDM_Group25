from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
from mpscenes.obstacles.urdf_obstacle import UrdfObstacle

import math
import numpy as np

import uuid

def theta_to_quaternion(theta): #Helper function to change euler angles to quaternions that URDF library objects uses
    theta = -theta          #Make negative so that positive orientation angle follow right hand rule
    qx = math.sin(theta/2)  #Only calculate yaw/z-axis rotation
    qy = 0.0
    qz = 0.0
    qw = math.cos(theta/2)
    return (qx, qy, qz, qw)


def create_env(env, obstacles_specs):
    """
    Takes an obstacle list with your custom format and inserts them into the
    environment using env.add_obstacle(). Supports:
    sphere, cube/box, cylinder, urdf, wall.
    """

    for obs in obstacles_specs:

        obs_type = obs["type"].lower()

        #----------------------------------------------------------------------
        # --- WALL HANDLING ---------------------------------------------------
        if obs_type == "wall":
            # Full-format wall
            pos = list(obs["position"])
            rgba = list(obs["rgba"])
            quaternion = theta_to_quaternion(obs["orientation"])
            content_dict = {
                "type": "box",
                "geometry": {
                    "position": pos,
                    "orientation" : quaternion,       #Rotation in quaternions
                    "width": obs["width"],
                    "length": obs["length"],
                    "height": obs["height"],
                }
            }

            if rgba:
                content_dict["rgba"] = rgba

            obstacle_obj = BoxObstacle(
                name=f"wall_{uuid.uuid4().hex[:6]}",
                content_dict=content_dict
            )

        # ---------------------------------------------------------------------
        # --- SPHERE ----------------------------------------------------------
        elif obs_type == "sphere":
            content_dict = {
                "type": "sphere",
                "geometry": {
                    "position": list(obs["position"]),
                    "radius": obs["radius"],
                }
            }
            if obs.get("rgba"):
                content_dict["rgba"] = list(obs["rgba"])

            obstacle_obj = SphereObstacle(
                name=f"sphere_{uuid.uuid4().hex[:6]}",
                content_dict=content_dict
            )

        # ---------------------------------------------------------------------
        # --- BOX / CUBE ------------------------------------------------------
        elif obs_type in ["box", "cube"]:
            content_dict = {
                "type": "box",
                "geometry": {
                    "position": list(obs["position"]),
                    "width": obs["width"],
                    "length": obs["length"],
                    "height": obs["height"],
                }
            }
            if obs.get("rgba"):
                content_dict["rgba"] = list(obs["rgba"])

            obstacle_obj = BoxObstacle(
                name=f"box_{uuid.uuid4().hex[:6]}",
                content_dict=content_dict
            )
            print(obstacle_obj)

        # ---------------------------------------------------------------------
        # --- CYLINDER --------------------------------------------------------
        elif obs_type == "cylinder":
            content_dict = {
                "type": "cylinder",
                "geometry": {
                    "position": list(obs["position"]),
                    "radius": obs["radius"],
                    "height": obs["height"],
                }
            }
            if obs.get("rgba"):
                content_dict["rgba"] = list(obs["rgba"])

            obstacle_obj = CylinderObstacle(
                name=f"cylinder_{uuid.uuid4().hex[:6]}",
                content_dict=content_dict
            )

        # ---------------------------------------------------------------------
        # --- URDF OBSTACLE ---------------------------------------------------
        elif obs_type == "urdf":
            content_dict = {
                "type": "urdf",
                "geometry": {
                    "position": list(obs["position"]),
                },
                "urdf": obs["urdf_path"],
            }

            obstacle_obj = UrdfObstacle(
                name=f"urdf_{uuid.uuid4().hex[:6]}",
                content_dict=content_dict
            )

        else:
            raise ValueError(f"Unknown obstacle type: {obs_type}")

        # Add to environment
        env.add_obstacle(obstacle_obj)
        print("Placed object: ", obs_type)
        print("Object specs: ", obs)
