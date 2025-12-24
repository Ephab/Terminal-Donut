import numpy as np
from math import sin, cos, radians
import os
import time

try:
    terminal_size = os.get_terminal_size()
    terminal_x = terminal_size.columns
    terminal_y = terminal_size.lines
except:
    print("Uh oh.. couldn't find a terminal 😔")
    quit(0)


def rotate_x(x, y, z, alpha):
    alpha = radians(alpha)
    
    rotation_about_x_axis_matrix = np.array([[1, 0, 0],[0, cos(alpha), -sin(alpha)],[0, sin(alpha), cos(alpha)]])
    result = np.matmul(rotation_about_x_axis_matrix, np.array([x, y, z]).reshape(3, 1))
    result = result.flatten()
    
    return result[0], result[1], result[2]
    
def rotate_y(x, y, z, beta):
    beta = radians(beta)
    
    rotation_about_y_axis_matrix = np.array([[cos(beta), 0, sin(beta)],[0, 1, 0],[-sin(beta), 0, cos(beta)]])
    result = np.matmul(rotation_about_y_axis_matrix, np.array([x,y,z]).reshape(3, 1))
    result = result.flatten()
    
    return result[0], result[1], result[2]

def rotate_z(x, y, z, gamma):
    gamma = radians(gamma)
    
    rotation_about_z_axis_matrix = np.array([[cos(gamma), -sin(gamma), 0],[sin(gamma), cos(gamma), 0],[0, 0, 1]])
    result = np.matmul(rotation_about_z_axis_matrix, np.array([x,y,z]).reshape(3, 1)) 
    result = result.flatten()
    
    return result[0], result[1], result[2]

def reflect_z(x, y, z):
    
    reflection_z_matrix = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
    result = np.matmul(reflection_z_matrix, np.array([x, y, z]))
    result = result.flatten()
    
    return result[0], result[1], result[2]
    

def rotate_point(x, y, z, alpha, beta, gamma, order=['x', 'y', 'z']):
    mapping = {'x': (rotate_x, alpha),
               'y': (rotate_y, beta),
               'z': (rotate_z, gamma)}
    
    x_prime, y_prime, z_prime = mapping[order[0]][0](x, y, z, mapping[order[0]][1])
    x_prime, y_prime, z_prime = mapping[order[1]][0](x_prime, y_prime, z_prime, mapping[order[1]][1])
    x_prime, y_prime, z_prime = mapping[order[2]][0](x_prime, y_prime, z_prime, mapping[order[2]][1])
    x_prime, y_prime, z_prime = reflect_z(x_prime, y_prime, z_prime)
    
    return x_prime, y_prime, z_prime
    
    
def generate_torus(r, R, theta, phi):
    theta = radians(theta)
    phi = radians(phi)
    
    x = (R + r*cos(theta))*cos(phi)
    y = r*sin(theta)
    z = -(R + r*cos(theta))*sin(phi)
    
    return (x, y, z)

def projection(x, y, z, k=6):
    # k is for avoiding / by 0 and for moving the camera back (kind of shrinking the x-y projection)
    x = x / (z + k)
    y = y / (z + k)
    depth = 1 / (z + k)
    
    return x, y, depth

def map_to_terminal(projected_x, projected_y, scale=50, t_x=terminal_x, t_y=terminal_y):
    centered_x = t_x / 2 
    centered_y = t_y / 2 #centered on terminal screen
    
    screen_x = centered_x + projected_x * scale
    screen_y = centered_y - projected_y * scale
    
    return screen_x, screen_y


def precompute_torus_points(R, r, degree_steps):
    l = []
    for theta in range(0, 360, degree_steps):
        for phi in range(0, 360, degree_steps):
            point = generate_torus(r, R, theta, phi)
            l.append(point)
            
    return l
    
def start():
    R, r = 2, 1
    degree_steps = 15
    
    screen_buffer = np.empty((abs(terminal_y), abs(terminal_x)), dtype='<U1')
    screen_buffer.fill(" ")
    
    a, b, c = 0, 0, 0
    points = precompute_torus_points(R, r, degree_steps) #much needed optimization (instead of running 24 times each loop or more)
    
    for i in range(10000):
        screen_buffer.fill(" ")
        
        a += 2; b += 1; c += 1
        
        for point in points:
            point = rotate_point(*point, alpha=a, beta=b, gamma=c, order=['y','x','z'])
            
            x, y, depth = projection(*point)
            
            buffer_x, buffer_y = map_to_terminal(x, y)
            
            if (0 <= buffer_x < terminal_x) and (0 <= buffer_y < terminal_y):
                screen_buffer[int(buffer_y), int(buffer_x)] = r':'
                        
        os.system("clear") # super expensive, should find something better
        # print("\x1b[H\x1b[J", end="")  # ansi better than clear? answer: no.
        for row in screen_buffer:
            print("".join(row))

start()