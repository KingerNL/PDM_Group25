import numpy as np
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
import uuid

def array_to_spec(env, obstacleArray):     #Gets 2D numpy array with: x, y and radius for each obstacle that should be made
                                        #Only can make circles (currently)
    #obstacles = []
    #obstacle_specs = []

    for obstacle in obstacleArray:
        x_centre = float(obstacle[0].tolist())
        y_centre = float(obstacle[1].tolist())
        radius = float(obstacle[2].tolist())
        print("type is: ", type(list((x_centre, y_centre, radius))))
        print("coordinates: ", x_centre, y_centre, radius)

        obstacle_spec = ({
            "type": "cylinder",
            "geometry": {
                "position": list((x_centre, y_centre, 0)),
                "radius": radius,
                "height": 2.0,
            }})
        obstacle_spec["rgba"] = list((0, 0, 0, 0.8))
        obstacle_obj = CylinderObstacle(
                name=f"cylinder_{uuid.uuid4().hex[:6]}",
                content_dict=obstacle_spec
            )
        env.add_obstacle(obstacle_obj)

    return env

#array_to_spec(testArray)

def static_scenario_1():
    obstacles = [
    {
        "type": "sphere",
        "position": (0.0, 4.0, 0),
        "orientation": 0.0,        
        "width": 0.0,
        "length": 0.0,
        "height": 2.0,
        "radius": 1.0,
        "rgba": (0.1, 0.1, 0.1, 0.9)
    },
    {
        "type": "cube",
        "position": (-3.0, 2.0, 0.0),
        "orientation": 0.0,        
        "width": 1.0,
        "length": 1.0,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (0.2, 0.4, 0.4, 0.9)
    },
    {
        "type": "wall",
        "position": (2.0, -2.0, 0.0),
        "orientation": np.pi/4,             #Orientation in radians (z-axis rotation)
        "width": 1.0,
        "length": 2.0,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (0.7, 0.2, 0.4, 0.9)
    }]
    return obstacles

def static_scenario_2():
    obstacles = [
        {
        "type": "cube",
        "position": (2.0, 2.0, 0.0),
        "orientation": 0.0,        
        "width": 1.0,
        "length": 1.0,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.8)
    },
        {
        "type": "cube",
        "position": (-3.0, 2.0, 0.0),
        "orientation": 0.0,        
        "width": 0.5,
        "length": 0.5,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (0.2, 0.4, 0.4, 0.9)
    },
        {
        "type": "cube",
        "position": (-3.0, -2.5, 0.0),
        "orientation": 0.0,        
        "width": 0.8,
        "length": 0.8,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (0.2, 0.4, 0.4, 0.9)
    },
        {
        "type": "cube",
        "position": (1.3, -2, 0.0),
        "orientation": 0.0,        
        "width": 1.4,
        "length": 1.4,
        "height": 2.0,
        "radius": None,                    # optional
        "rgba": (0.2, 0.4, 0.4, 0.9)
    }
    ]
    return obstacles

def dynamic_scenario_1():
    obstacles = []
    return obstacles

def racetrack():
    obstacles = [
    {
        "type": "wall",                     #Left wall at starting
        "position": (0, -1.5, 0.0),
        "orientation": np.pi/2,             #Orientation in radians (z-axis rotation)
        "width": 12,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.9)
    },
        {
        "type": "wall",                     #Right wall at starting
        "position": (0, 1.5, 0.0),
        "orientation": np.pi/2,             #Orientation in radians (z-axis rotation)
        "width": 6,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.9)
    },
    {
        "type": "wall",               #First corner wall, outer
        "position": (6.2, 2.3, 0.0),
        "orientation": 0,             #Orientation in radians (z-axis rotation)
        "width": 8,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1, 1, 1, 0.9)
    },
    {
        "type": "wall",             #First corner wall, inner
        "position": (3.2, 2.3, 0.0),
        "orientation": 0,             #Orientation in radians (z-axis rotation)
        "width": 2,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (0.7, 0.2, 0.4, 0.9)
    },
    {
        "type": "wall",                     #Second corner, outer
        "position": (3, 6.1, 0.0),
        "orientation": np.pi/2,             #Orientation in radians (z-axis rotation)
        "width": 6,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.9)
    },
    {
        "type": "wall",                     #Second corner, inner
        "position": (0.0, 3.1, 0.0),
        "orientation": np.pi/2,             #Orientation in radians (z-axis rotation)
        "width": 6,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.9)
    },
    {
        "type": "wall",                     #Third corner, outer
        "position": (0.0, 3.1, 0.0),
        "orientation": np.pi/2,             #Orientation in radians (z-axis rotation)
        "width": 6,
        "length": 0.4,
        "height": 1.5,
        "radius": None,                    # optional
        "rgba": (1.0, 1.0, 1.0, 0.9)
    },
    ]
    return obstacles