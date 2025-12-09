import numpy as np
import csv
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
import uuid

def add_obstacleArray_to_env(env, obstacleArray):  #Gets 2D numpy array with: x, y and radius for each obstacle that should be made
                                        #Only can make circles (currently)
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

    vertices = make_circle_boundingbox(obstacleArray, 0)
    generate_vertices_csv(vertices)

    return env

def make_circle_boundingbox(obstacleArray, margin=0):
    """
    Given a numpy array of circles (x_center, y_center, radius),
    return the corners of the bounding boxes as a flat array with a margin.
    
    Parameters:
        circles (np.ndarray): shape (n, 3), each row [x, y, r]
        margin (float): extra distance to extend the bounding box beyond the circle
    
    Returns:
        np.ndarray: shape (n, 8), corners for each circle
                    order: [top_left_x, top_left_y, top_right_x, top_right_y,
                            bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y]
    """
    x = obstacleArray[:, 0]
    y = obstacleArray[:, 1]
    r = obstacleArray[:, 2]

    xmin = x - r    
    xmax = x + r    
    ymin = y - r    
    ymax = y + r    

    print("top left (xmin) node x: ", xmin, "top left(ymin) node y: ", ymin)
    print("top right(xmin) node x: ", xmin, "top right(ymax) node y: ", ymax)
    print("bottom_right(xmax) node x: ", xmax, "bottom_right(ymax) node y: ", ymax)
    print("bottom_left(xman) node x: ", xmax, "bottom_left(ymin) node y: ", ymin)

    vertices = np.column_stack([
        xmin, ymin,     # top-left
        xmax, ymin,     # bottom-left
        xmax, ymax,     # bottom-right
        xmin, ymax      # top-right
    ])

    print(vertices)
    return vertices

def generate_vertices_csv(vertices, filename="obstacle_enviroment.csv"):
    print("making csv file")

    header = [
        "x1","y1","x2","y2","x3","y3","x4","y4"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(vertices)

    print(f"Saved {vertices.shape[0]} obstacles to {filename}")


    return

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