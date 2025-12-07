import numpy as np

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
    obstacles = []
    return obstacles

def dynamic_scenario_1():
    obstacles = []
    return obstacles

def racetrack():
    obstacles = []
    return obstacles