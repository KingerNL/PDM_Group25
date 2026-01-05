import numpy as np
import csv
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
import uuid
import pybullet as p


def add_obstacleArray_to_env(env, obstacleArray, offset, filename="", debug=False):  #Only can make cylinders (currently)                                            
    """
    Convert an array of circle coordinates (x, y, radius) to cylindrical obstacles to be placed into the enviroment of the robot
    Will also call the circle_boundigbox function and generate_vertices_csv function in order to
    make bounding boxes of the obstacles and save the coordinates of the box into an csv file

    Input:  env, UrdfEnv class: Instance of the UrdfEnv class which holds the initialized enviroment in which to place the obstacles
            obstacleArray, Numpy array: (n, 3) with (x, y, radius) coordinates of circles
            filename (optional), string: Path at which to save the generated .csv file

    Output: env, UrdfEnv class: Updated instance of the enviroment with cylindrical objects placed
    """
    for obstacle in obstacleArray:
        x_centre = float(obstacle[0].tolist())
        y_centre = float(obstacle[1].tolist())
        radius = float(obstacle[2].tolist())

        if debug == True:
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

    vertices = make_circle_boundingbox(obstacleArray, offset)
    #generate_vertices_csv(vertices)

    # if filename != "": generate_vertices_csv(vertices, filename=filename)
    # else: generate_vertices_csv(vertices)

    return env, vertices


def make_circle_boundingbox(obstacleArray, offset):

    """
    For array of coordinates of circles (x, y, radius)
    return corners of a square bounding box that fits the circle, with some margin if requested.

    Input:  Obstacle array, Numpy array: (n, 3) with (x, y, radius) coordinates of circles
            margin (optional), float: extra distance to keep inbetween boundig box and circle

    Output: vertices, Numpy array: (n, 8) with (Xmin, Ymin, Xmin, Ymax, Xmax, Ymax, Xmax, Ymin)
            corresponding to the (x, y) coordinates of each corner node
    """
    x = obstacleArray[:, 0]
    y = obstacleArray[:, 1]
    r = obstacleArray[:, 2]

    xmin = x - r - offset    
    xmax = x + r - offset   
    ymin = y - r - offset   
    ymax = y + r - offset  

    all_vertices = []

    # Loop over each obstacle and create its vertices
    for i in range(len(x)):
        vertices = [
            (xmin[i], ymin[i]),
            (xmax[i], ymin[i]),
            (xmax[i], ymax[i]),
            (xmin[i], ymax[i])
        ]
        all_vertices.append(vertices)

    return all_vertices


def generate_vertices_csv(vertices, filename="create_enviroment/obstacle_enviroment.csv"):
    """
    For array of coordinates of circles (x, y, radius)
    return corners of a square bounding box that fits the circle, with some margin if requested.

    Input:  vertices, Numpy array: (n, 8) with (Xmin, Ymin, Xmin, Ymax, Xmax, Ymax, Xmax, Ymin)
            filename (optional), string: Path at which to save the generated .csv file

    Output: None
    """     
    header = [
        "x1","y1","x2","y2","x3","y3","x4","y4"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(vertices)

    print(f"Saved {vertices.shape[0]} obstacles to {filename}")
    return

def add_visual_marker(position, radius=0.2, rgba=(0.0, 1.0, 0.0, 0.6)):
    """
    For a given position array/list (x, y, z), radius and rgba color value place a 
    non-coliding marker into the enviroment for visualization purposes, for example showing goal location

    Input:  position, Numpy array or list: of (x, y, z)
            radius (optional), float: radius of marker to be placed
            rgba (optional), list of floats: rgba color values of marker that will be placed

    Output: body_id, pybullet ID: ID of the visual object that was palced
    """
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=rgba
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=-1,  #No collision
        basePosition=position
    )
    return body_id

def remove_visual_marker(body_id):
    p.removeBody(body_id)
    return

def generate_random_obstacle_array(num_points, min_dist, max_radius, robot_pos, goal_position, robot_tol=0.5):
    """
    Generates a certain number of random obstacle cooridnates and radii
    Only valid objects are saved, 
    Object are valid if:
    1. Are not near the robot starting position
    2. Are some distance apart from eachother
    3. Are withing the enviroment boundaries
    If no solution can be found in some amount of tries, function will return the valid objects (which will be less than num_points)

    Input:  num_points, int: Amount of obstacle coordinates to generate
            min_dist, float: Minimum distance between objects
            max_radius, float: Maximum radius of object to be generated (minimum hardcoded to 1.0)
            robot_pos, list of float: (x,y) starting position of robot
            robot_tol (optional), float: Distance to keep between starting position and obstacles

    Output: obstacle_array, numpy array (num_points, 3): Coordinates and radii of randomly generated valid obstacles
    """
    obstacle_array = []
    tries = 0

    #Defining the square boundaries, default -15 to +15 in x and y
    limit = (-15.0, 15.0)
    
    np.random.seed(2) #Set seed for consistent output when replying

    while len(obstacle_array) < num_points:
        # Generate random x and y between -15 and 15
        x = np.random.uniform(limit[0], limit[1])
        y = np.random.uniform(limit[0], limit[1])
        radius = np.random.uniform(1.0, max_radius)

        is_safe = True

        #Check distance to robot starting position
        dist_to_robot = np.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
        if (dist_to_robot - radius) < robot_tol:
            is_safe = False
        
        #Check distance to robot goal position
        dist_to_robot = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)
        if (dist_to_robot - radius) < robot_tol:
            is_safe = False
            
        #Check distance from other objects
        if is_safe:
            for p in obstacle_array:
                prev_x, prev_y, prev_r = p
                center_dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if (center_dist - (radius + prev_r)) < min_dist:
                    is_safe = False
                    break
        
        if is_safe:
            obstacle_array.append([x, y, radius])

        #Safeguard to avoid searching for impossible solutions
        tries += 1
        if tries == num_points * 10:
            break
            
    return np.array(obstacle_array)


#Unused enviroment scenarios

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