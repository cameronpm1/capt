import numpy as np
from typing import Any, Dict, Optional, List, Tuple
import time

from util.util import gaussian_prob


class polarHistogram2D():

    def __init__(
            self,
            radius: float = 1,
            layers: int = 1,
            angle_sections: float = 36, 
            probability_tolerance: float = 0.05,
            distance_tolerance: float = 0.2,
    ):
        self.points = None
        self.radius = radius
        self.layers = 1 #always 1 layer for histogram
        self.layer_depth = self.radius
        self.probability_tol = probability_tolerance
        self.distance_tol = distance_tolerance

        self.sections = angle_sections
        self.range = 2*np.pi/self.sections
        self.histogram3D = np.zeros((self.sections,1,1,1)) #initialize the histogram
        self.reference_histogram3D = np.zeros((self.sections,1,3)) #center points of each bin on unit ball
        self.initialize_reference_histogram3D()
        self.histogram_calc_storage = None

    def convert_cartesian_to_polar(
            self,
            point: List[float],
    ):
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector
        theta2 = 0 #2d

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1

        dist = np.linalg.norm(point)

        return theta1,theta2,dist
    
    def convert_polar_to_bin(
            self,
            point: List[float],
    ) -> Tuple[int,int,int]:

        theta = int(point[0]//self.range)
        phi = 0
        layer = int(point[2]//self.layer_depth)

        if theta == self.sections:
            theta -= 1

        return theta, phi, layer
    
    def convert_cartesian_to_bin(
            self,
            point: List[float],
    ) -> Tuple[int,int,int]:
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1

        dist = np.linalg.norm(point)

        theta = int(theta1//self.range)
        phi = 0
        layer = int(dist//self.layer_depth)

        return theta, phi, layer

    def get_reference_point_from_bin(
            self,
            bin: List[int],
            layer: int = 0,
    ) -> List[float]:

        return self.reference_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer))
    
    def get_target_point_from_bin(
            self,
            bin: List[int],
            goal: List[float],
            layer: int = 0,
    ) -> Tuple[List[float], bool]:
        '''
        check if goal is inside chosen bin, 
        if goal is inside bin and within range -> return goal
        if goal is inside bin and within range -> return goal vector w/ appropriate distance from center
        '''

        theta1,theta2,dist = self.convert_cartesian_to_polar(goal)
        if int(theta1//self.range) == int(bin[0]) and 0 == int(bin[1]):
            if np.linalg.norm(goal) < (self.layer_depth * (0.5+layer)):
                return goal, True
            else:
                return goal/np.linalg.norm(goal) * (self.layer_depth * (0.5+layer)), False
        else:
            return self.reference_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer)), False
    
    def get_bin_from_index(
            self,
            bin: List[int],
    ) -> List[float]:
        
        return self.histogram3D[int(bin[0])][int(bin[1])]
    
    def reset_histogram(self) -> None:
        self.histogram3D[:] = 0

    def get_binary_histogram(self):
        '''
        Does not return binary histogram!!!
        '''        
        condensed_histogram = self.histogram3D[:,:,:,0].reshape(self.sections,self.layers)
        return condensed_histogram

    def input_points(
            self, 
            points: List[List[float]],
            points_min: int = 1,
    ) -> None:
        '''
        inputs points into histogram

        only stores closest point to origin in each bin
        '''
        self.points = points
        self.histogram3D[:] = 0
        a = 0

        for point in points:
            theta1,theta2,dist = self.convert_cartesian_to_polar(point)


            if dist > self.radius:
                next
            else:
                layer = int(dist//self.layer_depth)

                if self.histogram3D[int(theta1//self.range)][0][layer][0] == 0:
                    self.histogram3D[int(theta1//self.range)][0][layer][0] = dist
                elif self.histogram3D[int(theta1//self.range)][0][layer][0] > dist:
                        self.histogram3D[int(theta1//self.range)][0][layer][0] = dist


    def initialize_reference_histogram3D(self) -> None:
        '''
        create a polar historgram that contains the xyz 
        coordinates of the centerpoint of each bin w distance of 1
        '''
        for i in range(self.sections):
            for j in range(1):
                theta1 = i*self.range + self.range/2
                theta2 = 0
                
                x = np.cos(theta2)*np.cos(theta1)
                y = np.cos(theta2)*np.sin(theta1)
                z = np.sin(theta2)

                self.reference_histogram3D[i][j] = [x,y,z]

        

    

        


if __name__ == '__main__':

    histogram = polarHistogram2D(radius=13,angle_sections=4)

    points = [[1,-1,0]]

    histogram.input_points(points)
    a = histogram.get_histogram()
    print(a)
    '''
    test = histogram.refrerence_histogram3D

    x = []
    y = []
    z = []

    for i in range(histogram.sections):
        for j in range(histogram.sections):
            x.append([test[i][j][0]])
            y.append([test[i][j][1]])
            z.append([test[i][j][2]])

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.show()
    '''