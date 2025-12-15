import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import matplotlib.patches as pat

from collections import deque

from shapely.geometry import Polygon, Point

#========================================================================
# Dubins Path
#========================================================================

def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))

def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)**.5

class Dubins:
    """
    Class implementing a Dubins path planner with a constant turn radius.
    
    Attributes
    ----------
    radius : float
        The radius of the turn used in all the potential trajectories.
    point_separation : float
        The distance between points of the trajectories. More points increases
        the precision of the path but also augments the computation time of the
        colision check.

    Methods
    -------
    dubins_path
        Computes the shortest dubins path between two given points.
    generate_points_straight
        Turns a path into a set of point representing the trajectory, for
        dubins paths when the path is one of LSL, LSR, RSL, RSR.
    generate_points_curve
        Turns a path into a set of point representing the trajectory, for
        dubins paths when the path is one of RLR or LRL.
    find_center
        Compute the center of the circle described by a turn.
    lsl
        Dubins path with a left straight left trajectory.
    rsr
        Dubins path with a right straight right trajectory.
    rsl
        Dubins path with a right straight left trajectory.
    lsr
        Dubins path with a left straight right trajectory.
    lrl
        Dubins path with a left right left trajectory.
    rlr
        Dubins path with a right left right trajectory.
    """
    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def all_options(self, start, end, sort=False):
        """
        Computes all the possible Dubin's path and returns them, in the form
        of a list of tuples representing each option: (path_length,
        dubins_path, straight).

        Parameters
        ----------
        start :  tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the inital point.
        end : tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the final point.
        sort : bool
            If the list of option has to be sorted by decreasing cost or not.

        Returns
        -------
        The shortest list of points (x, y) linking the initial and final points
        given as input with only turns of a defined radius and straight line.

        """
        center_0_left = self.find_center(start, 'L')
        center_0_right = self.find_center(start, 'R')
        center_2_left = self.find_center(end, 'L')
        center_2_right = self.find_center(end, 'R')
        options = [self.lsl(start, end, center_0_left, center_2_left),
                   self.rsr(start, end, center_0_right, center_2_right),
                   self.rsl(start, end, center_0_right, center_2_left),
                   self.lsr(start, end, center_0_left, center_2_right),
                   self.rlr(start, end, center_0_right, center_2_right),
                   self.lrl(start, end, center_0_left, center_2_left)]
        if sort:
            options.sort(key=lambda x: x[0])
        return options

    def dubins_path(self, start, end):
        """
        Computes all the possible Dubin's path and returns the sequence of
        points representing the shortest option.

        Parameters
        ----------
        start :  tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the inital point.
        end : tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the final point.

        Returns
        -------
        The shortest list of points (x, y) linking the initial and final points
        given as input with only turns of a defined radius and straight line.
        In the form of a (2xn) numpy array.

        """
        options = self.all_options(start, end)
        dubins_path, straight = min(options, key=lambda x: x[0])[1:]
        return self.generate_points(start, end, dubins_path, straight)

    def generate_points(self, start, end, dubins_path, straight):
        """
        Transforms the dubins path in a succession of points in the 2D plane.

        Parameters
        ----------
        start: tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the inital point.
        end: tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the final point.
        dubins_path: tuple
            The representation of the dubins path in the form of a tuple
            containing:
                - the angle of the turn in the first circle, in rads.
                - the angle of the turn in the last circle, in rads.
                - the angle of the turn in the central circle, in rads, or the
                  length of the central segment if straight is true.
        straight: bool
            True if their is a central segment in the dubins path.

        Returns
        -------
        The shortest list of points (x, y) linking the initial and final points
        given as input with only turns of a defined radius and straight line.
        In the form of a (2xn) numpy array.

        """
        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        return self.generate_points_curve(start, end, dubins_path)

    def lsl(self, start, end, center_0, center_2):
        """
        Left-Straight-Left trajectories.
        First computes the poisition of the centers of the turns, and then uses
        the fact that the vector defined by the distance between the centers
        gives the direction and distance of the straight segment.

        .. image:: img/twoturnssame.svg

        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            True, to indicate that this path contains a straight segment.
        """
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (end[2]-alpha)%(2*np.pi)
        beta_0 = (alpha-start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, beta_2, straight_dist), True)

    def rsr(self, start, end, center_0, center_2):
        """
        Right-Straight-Right trajectories.
        First computes the poisition of the centers of the turns, and then uses
        the fact that the vector defined by the distance between the centers
        gives the direction and distance of the straight segment.
        
        .. image:: img/twoturnssame.svg

        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            True, to indicate that this path contains a straight segment.

        """
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (-end[2]+alpha)%(2*np.pi)
        beta_0 = (-alpha+start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, -beta_2, straight_dist), True)

    def rsl(self, start, end, center_0, center_2):
        """
        Right-Straight-Left trajectories.
        Because of the change in turn direction, it is a little more complex to
        compute than in the RSR or LSL cases. First computes the position of
        the centers of the turns, and then uses the rectangle triangle defined
        by the point between the two circles, the center point of one circle
        and the tangeancy point of this circle to compute the straight segment
        distance.

        .. image:: img/twoturnsopposite.svg

        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            True, to indicate that this path contains a straight segment.

        """
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = -(psia+alpha-start[2]-np.pi/2)%(2*np.pi)
        beta_2 = (np.pi+end[2]-np.pi/2-alpha-psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, beta_2, straight_dist), True)

    def lsr(self, start, end, center_0, center_2):
        """
        Left-Straight-Right trajectories.
        Because of the change in turn direction, it is a little more complex to
        compute than in the RSR or LSL cases. First computes the poisition of
        the centers of the turns, and then uses the rectangle triangle defined
        by the point between the two circles, the center point of one circle
        and the tangeancy point of this circle to compute the straight segment
        distance.

        .. image:: img/twoturnsopposite.svg
        
        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            True, to indicate that this path contains a straight segment.

            """
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = (psia-alpha-start[2]+np.pi/2)%(2*np.pi)
        beta_2 = (.5*np.pi-end[2]-alpha+psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, -beta_2, straight_dist), True)

    def lrl(self, start, end, center_0, center_2):
        """
        Left-right-Left trajectories.
        Using the isocele triangle made by the centers of the three circles,
        computes the required angles.

        .. image:: img/threeturns.svg

        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            False, to indicate that this path does not contain a straight part.
        """
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if 2*self.radius < dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = (psia-start[2]+np.pi/2+(np.pi-gamma)/2)%(2*np.pi)
        beta_1 = (-psia+np.pi/2+end[2]+(np.pi-gamma)/2)%(2*np.pi)
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len,
                (beta_0, beta_1, 2*np.pi-gamma),
                False)

    def rlr(self, start, end, center_0, center_2):
        """
        Right-left-right trajectories.
        Using the isocele triangle made by the centers of the three circles,
        computes the required angles.

        .. image:: img/threeturns.svg

        Parameters
        ----------
        start : tuple
            (x, y, psi) coordinates of the inital point.
        end : tuple
            (x, y, psi) coordinates of the final point.
        center_0 : tuple
            (x, y) coordinates of the center of the first turn.
        center_2 : tuple
            (x, y) coordinates of the center of the last turn.

        Returns
        -------
        total_len : float
            The total distance of this path.
        (beta_0, beta_2, straight_dist) : tuple
            The dubins path, i.e. the angle of the first turn, the angle of the
            last turn, and the length of the straight segment.
        straight : bool
            False, to indicate that this path does not contain a straight part.
        """
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if 2*self.radius < dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = -((-psia+(start[2]+np.pi/2)+(np.pi-gamma)/2)%(2*np.pi))
        beta_1 = -((psia+np.pi/2-end[2]+(np.pi-gamma)/2)%(2*np.pi))
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len,
                (beta_0, beta_1, 2*np.pi-gamma),
                False)


    def find_center(self, point, side):
        """
        Given an initial position, and the direction of the turn, computes the
        center of the circle with turn radius self.radius passing by the intial
        point.

        Parameters
        ----------
        point : tuple
            In the form (x, y, psi), with psi in radians.
            The representation of the inital point.
        side : Char
            Either 'L' to indicate a left turn, or 'R' for a right turn.

        Returns
        -------
        coordinates : 2x1 Array Like
            Coordinates of the center of the circle describing the turn.

        """
        assert side in 'LR'
        angle = point[2] + (np.pi/2 if side == 'L' else -np.pi/2)
        return np.array((point[0] + np.cos(angle)*self.radius,
                         point[1] + np.sin(angle)*self.radius))

    def generate_points_straight(self, start, end, path):
        """
        For the 4 first classes of dubins paths, containing in the middle a
        straight section.

        Parameters
        ----------
        start : tuple
            Start position in the form (x, y, psi).
        end : tuple
            End position in the form (x, y, psi).
        path : tuple
            The computed dubins path, a tuple containing:
                - the angle of the turn in the first circle, in rads
                - the angle of the turn in the last circle, in rads
                - the length of the straight line in between
            A negative angle means a right turn (antitrigonometric), and a
            positive angle represents a left turn.

        Returns
        -------
        The shortest list of points (x, y) linking the initial and final points
        given as input with only turns of a defined radius and straight line.
        In the form of a (2xn) numpy array.

        """
        total = self.radius*(abs(path[1])+abs(path[0]))+path[2] # Path length
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        # We first need to find the points where the straight segment starts
        if abs(path[0]) > 0:
            angle = start[2]+(abs(path[0])-np.pi/2)*np.sign(path[0])
            ini = center_0+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: ini = np.array(start[:2])
        # We then identify its end
        if abs(path[1]) > 0:
            angle = end[2]+(-abs(path[1])-np.pi/2)*np.sign(path[1])
            fin = center_2+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: fin = np.array(end[:2])
        dist_straight = dist(ini, fin)

        # We can now generate all the points with the desired precision
        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Straight segment
                coeff = (x-abs(path[0])*self.radius)/dist_straight
                points.append(coeff*fin + (1-coeff)*ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(self, start, end, path):
        """
        For the two last paths, where the trajectory is a succession of 3
        turns. First computing the position of the center of the central turn,
        then using the three circles to apply the angles given in the path
        argument.

        Parameters
        ----------
        start : tuple
            Start position in the form (x, y, psi).
        end : tuple
            End position in the form (x, y, psi).
        path : tuple
            The computed dubins path, a tuple containing:
                - the angle of the turn in the first circle, in rads
                - the angle of the turn in the last circle, in rads
                - the angle of the turn in the central circle, in rads
            A negative angle means a right turn (antitrigonometric), and a
            positive angle represents a left turn.

        Returns
        -------
        The shortest list of points (x, y) linking the initial and final points
        given as input with only turns of a defined radius. In the form of a
        (2xn) numpy array.

        """
        total = self.radius*(abs(path[1])+abs(path[0])+abs(path[2]))
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')
        intercenter = dist(center_0, center_2)
        center_1 = (center_0 + center_2)/2 +\
                   np.sign(path[0])*ortho((center_2-center_0)/intercenter)\
                    *(4*self.radius**2-(intercenter/2)**2)**.5
        psi_0 = np.arctan2((center_1 - center_0)[1],
                           (center_1 - center_0)[0])-np.pi

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Middle Turn
                angle = psi_0-np.sign(path[0])*(x/self.radius-abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1+self.radius*vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(self, reference, beta, center, x):
        """
        Returns the point located on the circle of center center and radius
        defined by the class, at the angle x.

        Parameters
        ----------
        reference : float
            Angular starting point, in radians.
        beta : float
            Used actually only to know the direction of the rotation, and hence
            to know if the path needs to be added or substracted from the
            reference angle.
        center : tuple
            (x, y) coordinates of the center of the circle from which we need a
            point on the circumference.
        x : float
            The lenght of the path on the circle.

        Returns
        -------
        The coordinates of the point on the circle, in the form of a tuple.
        """
        angle = reference[2]+((x/self.radius)-np.pi/2)*np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center+self.radius*vect

#========================================================================
# Static Environment
#========================================================================


"""
Implementation of the polygonal obstacles as well as of the walls
"""

class Obstacle:
    """
    Class implementing simple 2D polygonal obstacles.

    Attributes
    ----------
    points : list
        List of (x, y) coordinates in the frame of the environnement
        representing the obstacle.
    bounding_box : 4-tuple
        Coordinates of the lower left and upper right corners of the bounding
        box containing the obstacle.
    center : tuple
        Coordinates of the center of the bounding box.
    polygon : shapely.geometry.Polygon
        The polygon representing the obstacle.

    Methods
    -------
    plot
        Displays the polygon on screen.

    """

    def __init__(self, map_dimensions, size, nb_pts):
        self.center = np.array([np.random.rand()*map_dimensions[0],
                                np.random.rand()*map_dimensions[1]])
        # We use very simple convex polygons, generated with a radius
        # and randomly selected angles.
        angles = sorted((np.random.rand()*2*np.pi for _ in range(nb_pts)))
        self.points = \
            np.array([self.center +\
                      np.array([size*np.cos(angle), size*np.sin(angle)])\
                      for angle in angles])
        self.bounding_box = (min(self.points, key=lambda x: x[0])[0],
                             min(self.points, key=lambda x: x[1])[1],
                             max(self.points, key=lambda x: x[0])[0],
                             max(self.points, key=lambda x: x[1])[1])
        self.polygon = Polygon(self.points)

    def colides(self, x, y):
        """
        Checks if the given point is in the obstacle or not.
        """

        return self.polygon.contains(Point(x, y))

    def plot(self):
        """
        Draws the polygon on screen.
        """

        plt.gca().add_patch(pat.Polygon(self.points, color='black', fill=True))

class Wall:
    """
    Class implementing a wall with a moving hole in it

    Attributes
    ----------
    width : float
        The total width of the wall.
    bottom_y : float
        The position of the bottom of the wall.
    hole : float
        The position of the hole in the wall.
    thickness : float
        The thickness of the wall.
    speed : float
        The speed of the hole, if the obstacles are choosen to be dynamic.

    Methods
    -------
    colides
        Checks if a point is in the wall or not.
    plot
        Draws the wall on screen.
    visible
        Checks if the wall is in the field of view
    """

    def __init__(self, width, bottom_y, thickness, moving=False):
        self.width = width
        self.bottom_y = bottom_y
        self.hole = width*np.random.rand()
        self.thickness = thickness
        self.speed = (np.random.rand()-1/2)*2 if moving else 0

    def colides(self, x, y, time=0):
        """
        Checks if the given point is in the obstacle or not.
        """

        if time == 0:
            return (x < self.hole - self.width*.05 \
                    or x > self.hole + self.width*.05) \
                   and (self.bottom_y <= y <= self.bottom_y + self.thickness)

        hole = (self.hole + self.speed*time)%(self.width)
        return (x < hole - self.width*.05 or x > hole + self.width*.05)\
               and (self.bottom_y <= y <= self.bottom_y + self.thickness)

    def plot(self, time=0, x_scale=1, y_scale=1):
        """
        Draws the wall on screen.
        """

        hole = (self.hole+self.speed*time)%(self.width)
        plt.gca().add_patch(
            pat.Rectangle((0, self.bottom_y*y_scale),
                          hole-0.05*self.width, self.thickness*y_scale,
                          color='grey', fill=True))
        plt.gca().add_patch(
            pat.Rectangle(((hole+self.width*.05)*x_scale,
                           self.bottom_y*y_scale),
                          self.width*x_scale,
                          self.thickness*y_scale,
                          color='grey', fill=True))
        plt.gca().add_patch(
            pat.Rectangle((0, (self.bottom_y+self.thickness*.4)*y_scale),
                          hole - self.width*.1,
                          self.thickness*.2*y_scale,
                          color='black', fill=True))
        plt.gca().add_patch(
            pat.Rectangle(((hole+self.width*.1)*x_scale,
                           (self.bottom_y+self.thickness*.4)*y_scale),
                          (self.width)*x_scale,
                          self.thickness*.2*y_scale,
                          color='black', fill=True))

    def visible(self, view_top, view_bottom):
        """
        Checks if the wall is in the field of view
        """

        return self.bottom_y <= view_top \
               and self.bottom_y + self.thickness >= view_bottom
    
"""
The environment with static polygonal obstacles
"""

class StaticEnvironment:
    """
    Class implementing a very simple bounded 2D world, containing polygonal
    obstacles stored in an appropriate data structure for rapid access to close
    obstacles, even with a large amount of them.
    
    Attributes
    ----------
    dimensions : tuple
        (dim_x, dim_y) The x and y dimension of the rectangular world.
    obstacles : list
        List of obstacles, instances of the obstacle class.
    kdtree : KDTree
        The binary search tree used to have a rapid access to the obstacles,
        even with a large amount of them.
        
    Methods
    -------
    plot
        Draws the environnement using matplotlib.
    is_free
        Returns False if a point is within an obstacle or outside of the
        boundaries of the environnement.
    add_obstacle
        Manually adds an obstacle to the environment.
    """
    
    def __init__(self, dimensions, nb_obstacles=0):
        """
        Initialize the environment.
        
        Parameters
        ----------
        dimensions : tuple
            (dim_x, dim_y) The x and y dimension of the rectangular world.
        nb_obstacles : int, optional
            Number of random obstacles to generate. Default is 0 (no random obstacles).
        """
        self.dimensions = dimensions
        self.obstacles = []
        
        # Generate random obstacles if requested
        if nb_obstacles > 0:
            self.obstacles = [Obstacle(dimensions, 0.05*dimensions[0], 5)
                            for _ in range(nb_obstacles)]
        
        # Initialize KDTree (will be rebuilt when obstacles are added)
        self._rebuild_kdtree()
    
    def add_obstacle(self, obstacle):
        """
        Manually add an obstacle to the environment.
        
        Parameters
        ----------
        obstacle : Obstacle
            An instance of the Obstacle class to add to the environment.
        """
        self.obstacles.append(obstacle)
        self._rebuild_kdtree()
    
    def add_obstacle_at(self, center, radius, nb_vertices):
        """
        Create and add an obstacle at a specific location.
        
        Parameters
        ----------
        center : tuple
            (x, y) coordinates of the obstacle center.
        radius : float
            Radius of the obstacle.
        nb_vertices : int
            Number of vertices for the polygonal obstacle.
            
        Returns
        -------
        obstacle : Obstacle
            The created obstacle instance.
        """
        from shapely.geometry import Polygon, Point
        
        # Create obstacle with dummy initialization
        obstacle = Obstacle(self.dimensions, radius, nb_vertices)
        
        # Override with manual center and points
        obstacle.center = np.array(center)
        angles = sorted((np.random.rand()*2*np.pi for _ in range(nb_vertices)))
        obstacle.points = np.array([
            obstacle.center + np.array([radius*np.cos(angle), radius*np.sin(angle)])
            for angle in angles
        ])
        
        # Recalculate bounding box
        obstacle.bounding_box = (
            min(obstacle.points, key=lambda x: x[0])[0],
            min(obstacle.points, key=lambda x: x[1])[1],
            max(obstacle.points, key=lambda x: x[0])[0],
            max(obstacle.points, key=lambda x: x[1])[1]
        )
        
        # Recreate polygon
        obstacle.polygon = Polygon(obstacle.points)
        
        self.add_obstacle(obstacle)
        return obstacle
    
    def add_obstacle_with_vertices(self, vertices):
        """
        Create and add an obstacle at a specific location using given vertices.

        Parameters
        ----------
        vertices : list of tuple
            List of (x, y) coordinates for the obstacle's vertices (should be 4 for a quadrilateral).

        Returns
        -------
        obstacle : Obstacle
            The created obstacle instance.
        """
        from shapely.geometry import Polygon
        #print("Adding obstacle with vertices:", vertices)
        # Convert vertices to numpy array
        points = np.array(vertices)
        center = np.mean(points, axis=0)

        # Create obstacle with dummy initialization
        obstacle = Obstacle(self.dimensions, 1, len(points))  # size and nb_pts are placeholders
        obstacle.center = center
        obstacle.points = points

        # Recalculate bounding box
        obstacle.bounding_box = (
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1])
        )

        # Recreate polygon
        obstacle.polygon = Polygon(points)

        self.add_obstacle(obstacle)
        return obstacle


    def clear_obstacles(self):
        """
        Remove all obstacles from the environment.
        """
        self.obstacles = []
        self._rebuild_kdtree()
    
    def _rebuild_kdtree(self):
        """
        Rebuild the KDTree after obstacles have been added or removed.
        """
        if len(self.obstacles) > 0:
            self.kdtree = KDTree([obs.center for obs in self.obstacles])
        else:
            self.kdtree = None
    
    def plot(self, close=False, display=True, block=False):
        """
        Creates a figure and plots the environement on it.
        
        Parameters
        ----------
        close : bool
            If the plot needs to be automatically closed after the drawing.
        display : bool
            If the view pops up or not (used when generating many images)
        block : bool
            If True, block execution until plot window is closed (default False for RRT compatibility)
        """
        # Create a new figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # Plot obstacles
        for obstacle in self.obstacles:
            obstacle.plot()
        
        # Set limits and aspect
        ax.set_xlim(0, self.dimensions[0])
        ax.set_ylim(0, self.dimensions[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Environment with {len(self.obstacles)} Obstacles')
        ax.grid(True, alpha=0.3)
        
        if display and block:
            plt.show(block=True)
        elif display:
            plt.draw()
            plt.pause(0.001)
        
        if close:
            plt.close()
        
        return fig, ax
    
    def is_free(self, x, y, time=0):
        """
        Returns False if a point is within an obstacle or outside of the
        boundaries of the environnement.
        """
        if x < 0 or x > self.dimensions[0] or y < 0 or y > self.dimensions[1]:
            return False
        
        # If no obstacles, space is free
        if len(self.obstacles) == 0:
            return True
        
        for obstacle in self.close_obstacles(x, y, nb_obstacles=min(5, len(self.obstacles))):
            if obstacle.colides(x, y):
                return False
        return True
    
    def close_obstacles(self, x, y, nb_obstacles=1):
        """
        Returns the list of all the obstacles close enough to be considered.
        
        Parameters
        ----------
        x : float
            The x coordinate of the point requested
        y : float
            The y coordinate of the point requested
        nb_obstacles : int
            The number of obstacles to return, has to be less than the total
            number of obstacles of the environment.
            
        Note
        ----
        To be sure that this step actually does not remove any obstacle which
        could yield to a collision, the relation between the size of the
        obstacles and the considered radius for search must be verified:
            R_search > R_obs_Max
        With R_obs_Max the maximum distance between the center of an obstacle
        and one of its vertices.
        """
        if self.kdtree is None or len(self.obstacles) == 0:
            return []
        
        nb_obstacles = min(nb_obstacles, len(self.obstacles))
        indices = self.kdtree.query((x, y), nb_obstacles)[1]
        
        # Handle single obstacle case (query returns int instead of list)
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        
        return [self.obstacles[index] for index in indices]
    
    def random_free_space(self):
        """
        Returns a randomly selected point in the free space.
        """
        x = np.random.rand() * self.dimensions[0]
        y = np.random.rand() * self.dimensions[1]
        while not self.is_free(x, y):
            x = np.random.rand() * self.dimensions[0]
            y = np.random.rand() * self.dimensions[1]
        return x, y, np.random.rand() * np.pi * 2

#========================================================================
# RRT
#========================================================================

"""
Construction of the Rapidely Exploring Random Tree
"""

class Node:
    """
    Node of the rapidly exploring random tree.

    Attributes
    ----------
    destination_list : list
        The reachable nodes from the current one.
    position : tuple
        The position of the node.
    time : float
        The instant at which this node is reached.
    cost : float
        The cost needed to reach this node.
    """

    def __init__(self, position, time, cost):
        self.destination_list = []
        self.position = position
        self.time = time
        self.cost = cost

class Edge:
    """
    Edge of the rapidly exploring random tree.

    Attributes
    ----------
    node_from : tuple
        Id of the starting node of the edge.
    node_to : tuple
        Id of the end node of the edge.
    path : list
        The successive positions yielded by the local planner representing the
        path between the nodes.
    cost : float
        Cost associated to the transition between the two nodes.

    """

    def __init__(self, node_from, node_to, path, cost):
        self.node_from = node_from
        self.node_to = node_to
        self.path = deque(path)
        self.cost = cost

class RRT:
    """
    Class implementing a Rapidely Exploring Random Tree in two dimensions using
    dubins paths as an expansion method. The state space considered here is
    straightforward, as every state can be represented by a simple tuple made
    of three continuous variables: (x, y, psi)

    Attributes
    ----------
    nodes : dict
        Dictionnary containing all the nodes of the tree. The keys are hence
        simply the reached state, i.e. tuples of the form (x, y, psi).
    environment : Environment
        Instance of the Environment class.
    goal_rate : float
        The frequency at which the randomly selected node is choosen among
        the goal zone.
    precision : tuple
            The precision needed to stop the algorithm. In the form
            (delta_x, delta_y, delta_psi).
    goal : tuple
        The position of the goal (the center of the goal zone), in the form of
        a tuple (x, y, psi).
    root : tuple
        The position of the root of the tree, (the initial position of the
        vehicle), in the form of a tuple (x, y, psi).
    local_planner : Planner
        The planner used for the expansion of the tree, here it is a Dubins
        path planner.

    Methods
    -------
    in_goal_region
        Method helping to determine if a point is within a goal region or not.
    run
        Executes the algorithm with an empty graph, which needs to be
        initialized with the start position at least before.
    plot_tree
        Displays the RRT using matplotlib.
    select_options
        Explores the existing nodes of the tree to find the best option to grow
        from.
    """

    def __init__(self, environment, local_planner, precision=(5, 5, 1)):
        self.nodes = {}
        self.edges = {}
        self.environment = environment
        self.local_planner = local_planner
        self.goal = (0, 0, 0)
        self.root = (0, 0, 0)
        self.precision = precision

    def set_start(self, start):
        """
        Resets the graph, and sets the start node as root of the tree.

        Parameters
        ----------
        start: tuple
            The initial position (x, y, psi), used as root.
        """

        self.nodes = {}
        self.edges = {}
        self.nodes[start] = Node(start, 0, 0)
        self.root = start

    def run(self, goal, nb_iteration=100, goal_rate=.1, metric='local'):
        """
        Executes the algorithm with an empty graph, initialized with the start
        position at least.

        Parameters
        ----------
        goal : tuple
            The final requested position (x, y, psi).
        nb_iteration : int
            The number of maximal iterations (not using the number of nodes as
            potentialy the start is in a region of unavoidable collision).
        goal_rate : float
            The probability to expand towards the goal rather than towards a
            randomly selected sample.
        metric : string
            One of 'local' or 'euclidian'.
            The method used to select the closest node on the tree from which a
            path will be grown towards a sample.

        Notes
        -----
        It is not necessary to use several nodes to try and connect a sample to
        the existing graph; The closest node only could be choosen. The notion
        of "closest" can also be simpy the euclidian distance, which would make
        the computation faster and the code a simpler, this is why several
        metrics are available.
        """
        print("RRT: Starting run...")
        assert len(goal) == len(self.precision)
        self.goal = goal

        for _ in range(nb_iteration):
            # Select sample : either the goal, or a sample of free space
            if np.random.rand() > 1 - goal_rate:
                sample = goal
            else:
                sample = self.environment.random_free_space()
            # Find the closest Node in the tree, with the selected metric
            options = self.select_options(sample, 10, metric)

            # Now that all the options are sorted from the shortest to the
            # longest, we can try to connect one node after the other. We stop
            # after 10 in order to limit computations.
            for node, option in options:
                if option[0] == float('inf'):
                    break
                path = self.local_planner.generate_points(node,
                                                          sample,
                                                          option[1],
                                                          option[2])
                for i, point in enumerate(path):
                    if not self.environment.is_free(point[0],
                                                    point[1],
                                                    self.nodes[node].time+i):
                        break
                else:
                    #print(f"RRT: Added node at {sample} from {node}")
                    # Adding the node
                    # To compute the time, we use a constant speed of 1 m/s
                    # As the cost, we use the distance
                    self.nodes[sample] = Node(sample,
                                              self.nodes[node].time+option[0],
                                              self.nodes[node].cost+option[0])
                    self.nodes[node].destination_list.append(sample)
                    # Adding the Edge
                    self.edges[node, sample] = \
                        Edge(node, sample, path, option[0])
                    if self.in_goal_region(sample):
                        return
                    break

    def select_options(self, sample, nb_options, metric='local'):
        """
        Chooses the best nodes for the expansion of the tree, and returns
        them in a list ordered by increasing cost.

        Parameters
        ----------
        sample : tuple
            The (x, y, psi) coordinates of the node we wish to connect to the
            tree.
        nb_options : int
            The number of options requested.
        metric : str
            One of 'local', 'euclidian'. The euclidian metric is a lot faster
            but is also less precise and can't be used with an RRT star.

        Returns
        -------
        options : list
            Sorted list of the options, by increasing cost.
        """

        if metric == 'local':
            # The local planner is used to measure the real distance needed
            options = []
            for node in self.nodes:
                options.extend(
                    [(node, opt)\
                     for opt in self.local_planner.all_options(node, sample)])
            # sorted by cost
            options.sort(key=lambda x: x[1][0])
            options = options[:nb_options]
        else:
            # Euclidian distance as a metric
            options = [(node, dist(node, sample)) for node in self.nodes]
            options.sort(key=lambda x: x[1])
            options = options[:nb_options]
            new_opt = []
            for node, _ in options:
                db_options = self.local_planner.all_options(node, sample)
                new_opt.append((node, min(db_options, key=lambda x: x[0])))
            options = new_opt
        return options

    def in_goal_region(self, sample):
        """
        Method to determine if a point is within a goal region or not.

        Parameters
        ----------
        sample : tuple
            (x, y, psi) the position of the point which needs to be tested.
        """

        for i, value in enumerate(sample):
            if abs(self.goal[i]-value) > self.precision[i]:
                return False
        return True

    def plot(self, fig=None, ax=None, file_name='', close=False, nodes=False):
        """
        Displays the tree using matplotlib, on a currently open figure.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. If None, uses current figure.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, uses current axes.
        file_name : string
            The name of the file used to save an image of the tree.
        close : bool
            If the plot needs to be automatically closed after the drawing.
        nodes : bool
            If the nodes need to be displayed as well.
        """
        # Use provided axes or get current axes
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        
        # Plot nodes if requested
        if nodes and self.nodes:
            nodes_array = np.array(list(self.nodes.keys()))
            ax.scatter(nodes_array[:, 0], nodes_array[:, 1], 
                    c='blue', s=20, alpha=0.6, label='Tree Nodes')
        
        # Plot root (start) as green
        ax.scatter(self.root[0], self.root[1], 
                c='green', s=100, marker='o', 
                edgecolors='black', linewidths=2, 
                label='Start', zorder=5)
        
        # Plot goal as red
        ax.scatter(self.goal[0], self.goal[1], 
                c='red', s=100, marker='*', 
                edgecolors='black', linewidths=2, 
                label='Goal', zorder=5)
            

        # Plot edges/paths
        for _, val in self.edges.items():
            if val.path:
                # Convert to list and extract (x, y) for each point
                path_xy = [(p[0], p[1]) for p in val.path if len(p) >= 2]
                if len(path_xy) > 1:
                    path_xy = np.array(path_xy)
                    ax.plot(path_xy[:, 0], path_xy[:, 1], 
                            'b-', linewidth=0.5, alpha=0.5)


        # Highlight best path to goal   
        best_path, best_edges = self.get_best_path_to_goal()

        if best_path is not None:
            # Extract (x, y) for plotting
            best_path_xy = np.array([(p[0], p[1]) for p in best_path])
            ax.plot(best_path_xy[:, 0], best_path_xy[:, 1], 
                    'g-', linewidth=2, alpha=0.8, label='Best Path to Goal')
        if best_edges is not None:
            for edge in best_edges:
                path = np.array(edge.path)
                ax.plot(path[:, 0], path[:, 1], 
                    'g-', linewidth=2, alpha=0.8)

        # Save if filename provided
        if file_name:
            fig.savefig(file_name, dpi=300, bbox_inches='tight')

        # Close if requested
        if close:
            plt.close(fig)

    def get_path_to_goal(self):
        """
        Extracts the complete path from root to goal by backtracking through the tree.
        
        Returns
        -------
        path : list of tuples
            List of (x, y, theta) waypoints from root to goal.
        edges : list of Edge
            List of Edge objects forming the path from root to goal.
        """
        # Check if goal is in the tree
        if self.goal not in self.nodes:
            return None, None
        
        # Backtrack from goal to root
        path_nodes = []
        path_edges = []
        current = self.goal
        
        while current != self.root:
            path_nodes.insert(0, current)
            
            # Find parent node (the node that has current as a child)
            parent = None
            for node, node_obj in self.nodes.items():
                if current in node_obj.destination_list:
                    parent = node
                    break
            
            if parent is None:
                # Goal is not connected to root
                return None, None
            
            # Get the edge from parent to current
            edge = self.edges.get((parent, current))
            if edge:
                path_edges.insert(0, edge)
            
            current = parent
        
        # Add root at the beginning
        path_nodes.insert(0, self.root)
        
        # Construct full path with all waypoints
        full_path = [self.root]
        for edge in path_edges:
            if edge.path:
                pts = list(edge.path)
                for i in range(1, len(pts)):
                    prev = pts[i-1]
                    curr = pts[i]
                    angle = np.arctan2(curr[1] - prev[1], curr[0] - prev[0])
                    full_path.append((curr[0], curr[1], angle))
        
        return full_path, path_edges

    def select_best_edge(self):
        """
        Selects the best edge of the tree among the ones leaving from the root.
        Uses the number of children to determine the best option.

        Returns
        -------
        edge :Edge
            The best edge.
        """

        node = max([(child, self.children_count(child))\
                    for child in self.nodes[self.root].destination_list],
                   key=lambda x: x[1])[0]
        best_edge = self.edges[(self.root, node)]
        # we update the tree to remove all the other siblings of the old root
        for child in self.nodes[self.root].destination_list:
            if child == node:
                continue
            self.edges.pop((self.root, child))
            self.delete_all_children(child)
        self.nodes.pop(self.root)
        self.root = node
        return best_edge
    
    def get_best_path_to_goal(self):
        """
        Gets the path to goal if it exists, otherwise returns the path
        towards the node closest to the goal.
        
        Returns
        -------
        path : list of tuples
            List of (x, y, theta) waypoints.
        edges : list of Edge
            List of Edge objects forming the path.
        """
        # Try to get path to actual goal
        path, edges = self.get_path_to_goal()
        
        if path is not None:
            return path, edges
        
        # If goal not reached, find closest node to goal
        min_dist = float('inf')
        closest_node = None
        
        for node in self.nodes.keys():
            dist = np.sqrt((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        if closest_node is None:
            return None, None
        
        # Get path to closest node by temporarily setting it as goal
        original_goal = self.goal
        self.goal = closest_node
        path, edges = self.get_path_to_goal()
        self.goal = original_goal
        
        return path, edges

    def delete_all_children(self, node):
        """
        Removes all the nodes of the tree below the requested node.
        """

        if self.nodes[node].destination_list:
            for child in self.nodes[node].destination_list:
                self.edges.pop((node, child))
                self.delete_all_children(child)
        self.nodes.pop(node)

    def children_count(self, node):
        """
        Not optimal at all as it recounts a lot of the tree every time a path
        needs to b selected.
        """

        if not self.nodes[node].destination_list:
            return 0
        total = 0
        for child in self.nodes[node].destination_list:
            total += 1 + self.children_count(child)
        return total
            


def rrt_main(vertices):
        # # We initialize the planner with the turn radius and the desired distance between consecutive points
    # local_planner = Dubins(radius=0.6, point_separation=.5)

    # # We generate two points, x, y, psi
    # start = (0, 0, 2) # heading east
    # end = (20, 10, 3.141) # heading west

    # # We compute the path between them
    # path = local_planner.dubins_path(start, end)


    #----------------------------------------------------------------------------------
    # Create empty environment
    env_rrt = StaticEnvironment(dimensions=(20, 20))

    # Add obstacles at specific locations
    # env.add_obstacle_at(center=(5, 5), radius=1, nb_vertices=5)
    # env.add_obstacle_at(center=(10, 14), radius=1, nb_vertices=6)
    # env.add_obstacle_at(center=(15, 13), radius=1, nb_vertices=4)

    # print("vertices: " ,vertices)

    for vertice in vertices:
        env_rrt.add_obstacle_with_vertices(vertice)

    # Plot the environment - should now display!
    #env.plot(close=False, display=True)


    #----------------------------------------------------------------------------------
    # We initialize the tree with the environment


    rrt = RRT(env_rrt, precision=(1, 1, 0.5),local_planner=Dubins(radius=0.6, point_separation=0.1))

    # We select two random points in the free space as a start and final node
    # start = env.random_free_space()
    # end = env.random_free_space()

    start = (1, 1, 0) # heading east
    end = (18, 18, 3.141) # heading west

    # We initialize an empty tree
    rrt.set_start(start)
    
 
    # We run 100 iterations of growth
    rrt.run(end, nb_iteration=2000, goal_rate=0.08, metric='metric')

    # fig, ax = env_rrt.plot()
    # rrt.plot(fig=fig, ax=ax, nodes=True, file_name='rrt_path.png')
    best_path, best_edges = rrt.get_best_path_to_goal()

    # print("Best path waypoints:")
    # if best_path is not None:
    #     for waypoint in best_path:
    #         print(type(waypoint), waypoint)
    # else:
    #     print("No path to goal found.")

    # Keep the plot open
    # plt.show()
    return best_path