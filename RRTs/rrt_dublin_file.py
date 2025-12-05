"""
Dubins Path
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import LineString
from tqdm import tqdm


# class for PATH element
class PATH:
    def __init__(self, L, mode, x, y, yaw):
        self.L = L  # total path length [float]
        self.mode = mode  # type of each part of the path [string]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]


# utils
def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta


def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / math.pi / 2.0)


def LSL(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)

    if p_lsl < 0:
        return None, None, None, ["L", "S", "L"]
    else:
        p_lsl = math.sqrt(p_lsl)

    denominate = dist + sin_a - sin_b
    t_lsl = mod2pi(-alpha + math.atan2(cos_b - cos_a, denominate))
    q_lsl = mod2pi(beta - math.atan2(cos_b - cos_a, denominate))

    return t_lsl, p_lsl, q_lsl, ["L", "S", "L"]


def RSR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)

    if p_rsr < 0:
        return None, None, None, ["R", "S", "R"]
    else:
        p_rsr = math.sqrt(p_rsr)

    denominate = dist - sin_a + sin_b
    t_rsr = mod2pi(alpha - math.atan2(cos_a - cos_b, denominate))
    q_rsr = mod2pi(-beta + math.atan2(cos_a - cos_b, denominate))

    return t_rsr, p_rsr, q_rsr, ["R", "S", "R"]


def LSR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)

    if p_lsr < 0:
        return None, None, None, ["L", "S", "R"]
    else:
        p_lsr = math.sqrt(p_lsr)

    rec = math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr)
    t_lsr = mod2pi(-alpha + rec)
    q_lsr = mod2pi(-mod2pi(beta) + rec)

    return t_lsr, p_lsr, q_lsr, ["L", "S", "R"]


def RSL(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)

    if p_rsl < 0:
        return None, None, None, ["R", "S", "L"]
    else:
        p_rsl = math.sqrt(p_rsl)

    rec = math.atan2(cos_a + cos_b, dist - sin_a - sin_b) - math.atan2(2.0, p_rsl)
    t_rsl = mod2pi(alpha - rec)
    q_rsl = mod2pi(beta - rec)

    return t_rsl, p_rsl, q_rsl, ["R", "S", "L"]


def RLR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    rec = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["R", "L", "R"]

    p_rlr = mod2pi(2 * math.pi - math.acos(rec))
    t_rlr = mod2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b) + mod2pi(p_rlr / 2.0))
    q_rlr = mod2pi(alpha - beta - t_rlr + mod2pi(p_rlr))

    return t_rlr, p_rlr, q_rlr, ["R", "L", "R"]


def LRL(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    rec = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_b - sin_a)) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["L", "R", "L"]

    p_lrl = mod2pi(2 * math.pi - math.acos(rec))
    t_lrl = mod2pi(-alpha - math.atan2(cos_a - cos_b, dist + sin_a - sin_b) + p_lrl / 2.0)
    q_lrl = mod2pi(mod2pi(beta) - alpha - t_lrl + mod2pi(p_lrl))

    return t_lrl, p_lrl, q_lrl, ["L", "R", "L"]


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc
        if m == "L":
            ldy = (1.0 - math.cos(l)) / maxc
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)

        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions


def generate_local_course(L, lengths, mode, maxc, step):
    point_num = int(L / step) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step
    else:
        d = -step

    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step
        else:
            d = -step

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = \
                interpolate(ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = \
            interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)

    if len(px) <= 1:
        return [], [], [], []

    # remove unused data
    while len(px) >= 1 and px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def planning_from_origin(gx, gy, gyaw, curv, step):
    D = math.hypot(gx, gy)
    d = D * curv

    theta = mod2pi(math.atan2(gy, gx))
    alpha = mod2pi(-theta)
    beta = mod2pi(gyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    path_list = []
    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)

        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        lengths = [t, p, q]
        x_list, y_list, yaw_list, directions = generate_local_course(
            sum(lengths), lengths, mode, curv, step)
        path_list.append([x_list, y_list, yaw_list, mode, cost])

    return path_list


def calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, curv, step=0.1):
    goal_x = gx - sx
    goal_y = gy - sy

    l_rot = Rot.from_euler('z', syaw).as_matrix()[0:2, 0:2]
    le_xy = np.stack([goal_x, goal_y]).T @ l_rot
    le_yaw = gyaw - syaw

    possible_paths = []
    path_list = planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, curv, step)
    for path in path_list:
        lp_x, lp_y, lp_yaw, mode, lengths = path
        rot = Rot.from_euler('z', -syaw).as_matrix()[0:2, 0:2]
        converted_xy = np.stack([lp_x, lp_y]).T @ rot
        x_list = converted_xy[:, 0] + sx
        y_list = converted_xy[:, 1] + sy
        yaw_list = [pi_2_pi(i_yaw + syaw) for i_yaw in lp_yaw]
        possible_paths.append(PATH(lengths, mode, x_list, y_list, yaw_list))
    return possible_paths

def plan_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc, step=0.2):
    paths = []
    for curve_rate in maxc:
        paths += calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, curve_rate, step)
    if paths is None:
        return None # could not generate any path

    # search minimum cost path
    linestring_list = []
    for path in paths:
        x, y = np.array(path.x), np.array(path.y)
        coord = np.vstack((x, y)).T
        linestring_list.append(LineString(coord))
    return sorted(linestring_list, key=lambda x: x.length)




class TreeNode: # Tree Node class that RRT uses
    def __init__(self, point, yaw):
        self.point = point # This should be shapely PointObject, containing coordinate informatioj
        self.yaw = yaw # Orientation of vehicle

        self.cost = 0 # Initial cost of zero
        self.parent = None # Children, list of nodes
        self.path_to_parent = None # Path from parent, list of paths
        self.children = [] # Children, only used to draw the entire RRT tree




## RRT/ RRT* implementation for global motion planner ##
def RRT(N_iter, scenario, step_size=float('inf'), dist_tolerance=1, star=True, non_holonomic=True, force_return_tree=False, backwards=True):
    # Initialise start and goal node
    start_Node, goal_Node = TreeNode(scenario.start[0], scenario.start[1]), TreeNode(scenario.goal[0], scenario.goal[1])
    for n in tqdm(range(N_iter)): # Max N_iter iterations
        if n == 1: # First iteration, pick the goal node to try if an easy path already exists
            sampled_Node = goal_Node
        elif star and n == N_iter - 1: # For RRT*, pick the goal node as the last node to improve convergence chances
            sampled_Node = goal_Node
        elif not star and np.random.random_sample() < 0.05: # For normal RRT, have a chance of picking the goal node as the sampled node
            sampled_Node = goal_Node
        else:  # Ot herwise, randomly sample a point in the environment
            sampled_Node = TreeNode(rand_coords(scenario.width, scenario.height), np.deg2rad(np.random.randint(0, 360,1))[0])
           
            # Random chance to orient point to goal, only if sampled point is not the goal
            if not sampled_Node.point.equals(goal_Node.point) and np.random.random_sample() < 0.1:
                angle_to_goal = (np.degrees(np.arctan2((goal_Node.point.y - sampled_Node.point.y), (goal_Node.point.x - sampled_Node.point.x))) + 360) % 360
                sampled_Node.yaw = angle_to_goal

        # If the sampled point collides with obstacles
        if not scenario.collision_free(sampled_Node.point):
            continue # Continue to next iteration

        # Find the closest node to sampled point
        parent_Node, path_to_parent, _ = find_closest_Node(scenario, start_Node, sampled_Node, non_holonomic, backwards)

        if parent_Node is not None and not parent_Node.point.equals(sampled_Node.point):  # If a nearest node is found     
            if path_to_parent.length > step_size: # In the case that this parent node is not within the step size
                # Find a new point on the connecting line that is within the radius
                new_Node_inradius = TreeNode(path_to_parent.interpolate(step_size), parent_Node.yaw)

                # Update the sampled Node to this new point
                sampled_Node = new_Node_inradius
                        
            # Update the sampled_Node cost
            sampled_Node.cost = parent_Node.cost + path_to_parent.length 

            if star: # If RRT* is used, the sampled Node will be connected to the best nearby node
                radius = 20  # Within a certain radius, find nearby points
                Nodes_near_sample = find_nearby_nodes(start_Node, sampled_Node, radius, nearby_Nodes=[])

                if Nodes_near_sample: # If there are nearby nodes
                    min_cost = sampled_Node.cost # First set the upper bound of cost (current cost of sampled Node)
                    
                    for node in Nodes_near_sample: # For each nearby node, check if it is better to connect the sampled node to it
                        # Estimate cost by using distance and difference in angle, avoids redrawing new connectors for each node
                        cost_estimate = sampled_Node.point.distance(node.point) + min(abs(sampled_Node.yaw - node.yaw), 360 - abs(sampled_Node.yaw - node.yaw))
                        cost_via_node = node.cost + cost_estimate

                        if cost_via_node < min_cost: # If this cost is less, update the parent Node and min cost
                            parent_Node, min_cost = node, cost_via_node 
    
            # Create the path to parent and check if it is collision-free
            path_to_parent = create_connector(parent_Node, sampled_Node, scenario, non_holonomic, backwards)
            if path_to_parent is None:
                continue

            # Update the sampled_Node cost, in case there is a new path generated
            sampled_Node.cost = parent_Node.cost + path_to_parent.length 

            # Update parent node with children, and sampled Node with parent
            parent_Node.children.append(sampled_Node)
            sampled_Node.parent = parent_Node 
            sampled_Node.path_to_parent = path_to_parent

            # Rewiring the tree for RRT*, connect surrounding nodes to new sampled node if it is better
            if star and Nodes_near_sample: 
                for node in Nodes_near_sample: 
                    cost_estimate = sampled_Node.point.distance(node.point) + min(abs(sampled_Node.yaw - node.yaw), 360 - abs(sampled_Node.yaw - node.yaw))
                    cost_via_sampled_Node = sampled_Node.cost + cost_estimate # Estimate the cost to go via the sampled_Node

                    if cost_via_sampled_Node < node.cost: # The node is expected to have a lower cost if it is rewired to the sampled_Node
                        # Create a path from sampled_Node to node
                        path_to_node = create_connector(sampled_Node, node, scenario, non_holonomic, backwards)
                        if path_to_node is None: # if the path cannot be found or it collides with environment
                            continue # continue to next node

                        node_updated_cost = sampled_Node.cost + path_to_node.length # Get the new cost
                        if node_updated_cost < node.cost: # If the new path is indeed better
                            node.parent.children.remove(node) # Firstly destroy the parent connection of the node
                            node.parent = sampled_Node # Then update the node with its new parent (sampled node)
                            node.path_to_parent = path_to_node
                            sampled_Node.children.append(node) # add node as child of sampled_Node

                            node_cost_difference = node.cost - node_updated_cost # Find the cost difference
                            update_children_cost(node, node_cost_difference) # Update children with new cost

            # Early stop if normal RRT is used, as once the goal is reached the path won't change
            if not star and sampled_Node.point.distance(goal_Node.point) < dist_tolerance:
                final_path = extract_path(sampled_Node)
                total_tree = extract_all_edges(start_Node)
                
                # Update path and tree attributes of class
                scenario.set_path(final_path)
                scenario.set_totaltree(total_tree)

                print(f"\nRRT finished within {n} iterations")
                return
    
    # Find the nodes near the goal
    Nodes_near_goal = find_nearby_nodes(start_Node, goal_Node, dist_tolerance)

    if Nodes_near_goal:
        # Else RRT* has found a path, find the shortest one if there are several
        Node_min_cost = min(Nodes_near_goal, key=lambda x: (x.cost) * (goal_Node.point.distance(x.point) + min(abs(goal_Node.yaw - x.yaw), 360 - abs(goal_Node.yaw - x.yaw))))
        shortest_path = extract_path(Node_min_cost)

        # Finally set final path and 'total' tree containing all edges
        final_path = shortest_path 
        total_tree = extract_all_edges(start_Node)

        # Update class attributes
        scenario.set_path(final_path)
        scenario.set_totaltree(total_tree)

        # RRT* is done
        print(f"\nRRT* finished within {n+1} iterations")
        return

    else: # If it does not converge, there will be no Nodes near the goal
        if force_return_tree: # Force plot the tree, regardless whether it converges
            print("Force plotted the entire tree, convergence not guaranteed")
            scenario.set_totaltree(extract_all_edges(start_Node))
            return 
        # Raise exception if not force plotting
        raise Exception("\nRRT could not find a suitable path within the given number of iterations. Please try again.\nIf it consistently fails to complete, increase the number of iterations.")
    

def rand_coords(width, height): # Generate random coordinates within bounds of environment
    x = np.random.randint(0,width*100,1) / 100
    y = np.random.randint(0,height*100,1) / 100
    return Point(x, y)


def find_closest_Node(scenario, start_Node, new_Node, non_holonomic, backwards, min_length=float('inf')): # Find closest Node in Tree
    nearest_Node, shortest_path = None, None # Initalise nearest Node and shortest path as None

    # Recursively check all children
    if start_Node.children: # If Node has children
        for child in start_Node.children: # Iterate over the children nodes
            temp_Node, temp_path, temp_length = find_closest_Node(scenario, child, new_Node, non_holonomic, backwards, min_length=min_length)
            if temp_Node is None: # If the node does not exist
                return nearest_Node, shortest_path, min_length # Return current best Node and path
            if temp_length < min_length:
                nearest_Node, shortest_path, min_length = temp_Node, temp_path, temp_length
    
    # Either there are no children, or the recursive search is done (following code will be reached) #

    # Create connector that connects the two Nodes, returns None if the connector collides with environment
    connect_line = create_connector(start_Node, new_Node, scenario, non_holonomic, backwards)
    if connect_line is None:
        return nearest_Node, shortest_path, min_length # Return current best Nodes and path
    
    length = connect_line.length # Find length of connecting line

    # If this length is less than the saved min, we can once again update the nearest node
    if length < min_length:
        nearest_Node = start_Node # set nearest node
        shortest_path = connect_line # define the shortest path
        min_length = length # then update minimum length
    
    return nearest_Node, shortest_path, min_length


def extract_all_edges(start_Node, total_tree=[]): # Extract all edges from tree
    if start_Node.children: # Iterate over all the children of the start_Node
        for child in start_Node.children: # For each child
            total_tree.append(child.path_to_parent) # Append path to total_tree
            extract_all_edges(child, total_tree) # Recursively repeat over all its children
    return total_tree


def find_nearby_nodes(start_Node, goal_Node, tol, nearby_Nodes=[]): # Find Nodes that are near the goal if several paths exist
    if start_Node.children: # Iterate over all the children of the start_Node
        for child in start_Node.children: # For each child
            if child.point.distance(goal_Node.point) < tol: # If the child is within the tolerated distance of goal node
                nearby_Nodes.append(child) # Add the node to the nearby_Nodes list
            find_nearby_nodes(child, goal_Node, tol, nearby_Nodes) # Recursively repeat over all its children
    return nearby_Nodes


def update_children_cost(start_Node, cost_difference):
    if start_Node.children: # Iterate over all the children of the start_Node
        for child in start_Node.children: # For each child
            child.cost = child.cost - cost_difference # Update the cost of the child
    return


def extract_path(final_Node, path=[]): # Extract only final path from tree
    if final_Node.parent is not None: # As long as there is a parent
        path.append(final_Node.path_to_parent) # Append the path_to_parent to final path
        extract_path(final_Node.parent, path) # Recursively repeat one layer up
    return path


def create_connector(Node1, Node2, scenario, non_holonomic, backwards):
    # If linear connectors are desired, non_holonomic can be set to False
    if not non_holonomic:
        return LineString([Node1.point, Node2.point])
    maxc = scenario.max_curvature # Curvature
    sx, sy, syaw = Node1.point.x, Node1.point.y , round(np.radians(round(Node1.yaw,2)),2) # Coordinates and orientation of Node 1
    gx, gy, gyaw = Node2.point.x, Node2.point.y, round(np.radians(round(Node2.yaw,2)),2) # Coordinates and orientatio of Node 2
    if gx != sx and gy != sy: # Checks both node to connect arent the same node
        if not backwards:
            # Return list of possible connecting curves from Reeds-Schepp
            connect_line_list = plan_dubins_path(sx, sy, syaw, gx, gy, gyaw, maxc)
            if connect_line_list is not None: # If list is not empty
                for connect_line in connect_line_list:
                    if scenario.collision_free(connect_line): # If the line does not collide with the environment
                        return connect_line
            return None # If it does collide with the environment, return None
        else:
            # Return list of possible connecting curves from Reeds-Schepp
            connect_line_list = reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc)
            if connect_line_list is not None: # If list is not empty
                for connect_line in connect_line_list:
                    if scenario.collision_free(connect_line): # If the line does not collide with the environment
                        return connect_line
            return None # If it does collide with the environment, return None
        


import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
import math
from shapely import affinity
from bin import Controller
from bin import draw
import time as tt
from shapely.geometry import MultiLineString
from bin.csv_utils import csv_keys



def run_sim(simple_Scenario, name, animate=True):
    cx, cy, cyaw, reversing = simple_Scenario.read_csv(csv_keys[name], set_path=True)
    cyaw = np.deg2rad([-(360-i) if i>180 else i for i in cyaw])
    #cyaw = np.deg2rad(cyaw)
    sp = Controller.calc_speed_profile(cx, Controller.P.target_speed, reversing)
    ref_path = Controller.PATH(cx, cy, cyaw)
    node = Controller.Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    time2 = 0.0
    x = [node.x]
    y = [node.y]
    yaw = [node.yaw]
    v = [node.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]

    delta_opt, a_opt = None, None
    a_exc, delta_exc = 0.0, 0.0
    start_time = tt.time()
    if animate:
        manager=plt.get_current_fig_manager()
        manager.full_screen_toggle()
    while time2 < Controller.P.time_max:
        z_ref, target_ind = Controller.calc_ref_trajectory_in_T_step(node, ref_path, sp)

        z0 = [node.x, node.y, node.v, node.yaw]

        a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = Controller.linear_mpc_control(z_ref, z0, a_opt, delta_opt)

        if delta_opt is not None:
            delta_exc, a_exc = delta_opt[0], a_opt[0]

        node.update(a_exc, delta_exc, 1.0)
        time2 += Controller.P.dt

        x.append(node.x)
        y.append(node.y)
        yaw.append(node.yaw)
        v.append(node.v)
        t.append(time2)
        d.append(delta_exc)
        a.append(a_exc)

        dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

        if dist < Controller.P.dist_stop and \
                abs(node.v) < Controller.P.speed_stop:
            break

        dy = (node.yaw - yaw[-2]) / (node.v * Controller.P.dt)
        steer = Controller.pi_2_pi(-math.atan(Controller.P.WB * dy))
        
        if animate:
            plt.cla()

            draw.draw_car(node.x, node.y, node.yaw, steer, Controller.P)
            for obstacle in simple_Scenario.obstacles:
                obstacle = affinity.translate(obstacle, yoff = (simple_Scenario.vehicle_length/2 - Controller.P.RB))
                plt.gca().add_patch(matplotlib.patches.Polygon(obstacle.exterior.coords, color="grey"))

            if simple_Scenario.start is not None and simple_Scenario.goal is not None:
                if simple_Scenario.vehicle_length != 0 and simple_Scenario.vehicle_width != 0: # If the vehicle size has been set, draw start and goal as vehicle
                #  plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.start[0].x - Controller.P.RB, simple_Scenario.start[0].y - simple_Scenario.vehicle_width / 2),
                #                                              simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.start[1], color='red', alpha=0.8, label='Start', rotation_point='center'))
                    plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.start[0].x + simple_Scenario.vehicle_width / 2, simple_Scenario.start[0].y - Controller.P.RB),
                                                                simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.start[1], color='red', alpha=0.8, label='Start', rotation_point='xy'))
                    plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.goal[0].x + simple_Scenario.vehicle_width / 2, simple_Scenario.goal[0].y - Controller.P.RB), 
                                                                simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.goal[1], color='green', alpha=0.8, label='Goal', rotation_point='xy'))
                else: # Draw start and goal as points
                    plt.scatter(simple_Scenario.start[0].x, simple_Scenario.start[0].y, s=50, c='g', marker='o', label='Start')
                    plt.scatter(simple_Scenario.goal[0].x, simple_Scenario.goal[0].y, s=60, c='r', marker='*', label='Goal')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event:
                                        [exit(0) if event.key == 'escape' else None])

            if x_opt is not None:
                plt.plot(x_opt, y_opt, color='darkviolet', marker='*')

            plt.plot(cx, cy, color='gray', label='Planned Path')
            plt.plot(x, y, '-b', label='Simulated Path')
            plt.plot(cx[target_ind], cy[target_ind])

            plt.legend()
            plt.axis("equal")
            plt.title(name)
            plt.pause(0.001)
            
    end_time = tt.time()
    print(f"Simulation Time: {-(start_time - end_time)}")

    path = MultiLineString(simple_Scenario.path)
    print(f"Length of Planned Trajectory: {path.length}")

    simulated_path_points = list(zip(x,y))
    simulated_path = MultiLineString([[simulated_path_points[i], simulated_path_points[i+1]] for i in range(len(simulated_path_points) - 1)])
    print(f"Length of Simulated Trajectory: {simulated_path.length}")
    
    if not animate:
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(900*px, 900*px))

        # Set size of plot with correct aspect
        ax.set_aspect(aspect=1)
        
        # Set boundaries for drawing scenario
        offset =  simple_Scenario.vehicle_length/2 - Controller.P.RB
        plt.xlim([0, simple_Scenario.width])
        plt.ylim([0 + offset, simple_Scenario.height + offset])

        for obstacle in simple_Scenario.obstacles:
                obstacle = affinity.translate(obstacle, yoff = (simple_Scenario.vehicle_length/2 - Controller.P.RB))
                plt.gca().add_patch(matplotlib.patches.Polygon(obstacle.exterior.coords, color="grey"))
        
        # Draw start and goal
        if simple_Scenario.start is not None and simple_Scenario.goal is not None:
                if simple_Scenario.vehicle_length != 0 and simple_Scenario.vehicle_width != 0: # If the vehicle size has been set, draw start and goal as vehicle
                #  plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.start[0].x - Controller.P.RB, simple_Scenario.start[0].y - simple_Scenario.vehicle_width / 2),
                #                                              simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.start[1], color='red', alpha=0.8, label='Start', rotation_point='center'))
                    plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.start[0].x + simple_Scenario.vehicle_width / 2, simple_Scenario.start[0].y - Controller.P.RB),
                                                                simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.start[1], color='red', alpha=0.8, label='Start', rotation_point='xy'))
                    plt.gca().add_patch(matplotlib.patches.Rectangle((simple_Scenario.goal[0].x + simple_Scenario.vehicle_width / 2, simple_Scenario.goal[0].y - Controller.P.RB), 
                                                                simple_Scenario.vehicle_length, simple_Scenario.vehicle_width, simple_Scenario.goal[1], color='green', alpha=0.8, label='Goal', rotation_point='xy'))
                else: # Draw start and goal as points
                    plt.scatter(simple_Scenario.start[0].x, simple_Scenario.start[0].y, s=50, c='g', marker='o', label='Start')
                    plt.scatter(simple_Scenario.goal[0].x, simple_Scenario.goal[0].y, s=60, c='r', marker='*', label='Goal')

        # Draw path
        plt.plot(cx, cy, color='gray', label='Planned Path')
        plt.plot(x, y, '-b', label='Simulated Path')
        plt.title(name)     
        plt.legend()
        plt.show()



run_sim(TestScenario, "Street_Scenario_ReedsShepp", animate=True)
