import habitat_sim
from PIL import Image

from habitat.utils.visualizations import maps

import quaternion
import numpy as np

def get_scenes(path):
    files = open(path, "r")
    scenes = []
    for file in files.readlines():
        scenes.append(file.strip())
    return scenes

def get_random_point(sim):
    return sim.pathfinder.get_random_navigable_point()

def get_position(sim):
    agent_state = sim.agents[0].get_state()
    return agent_state.position

def get_orientation(sim):
    agent_state = sim.agents[0].get_state()
    return agent_state.rotation

def set_position(sim, position):
    agent_state = sim.agents[0].get_state()
    agent_state.position = position
    sim.agents[0].set_state(agent_state)

def set_orientation(sim, orientation):
    agent_state = sim.agents[0].get_state()
    agent_state.rotation = orientation
    sim.agents[0].set_state(agent_state)

def path_exists(sim, start, end):
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    return sim.pathfinder.find_path(path) and path.geodesic_distance >= 0.1

def to_map_coord(sim, world_point, grid_dimensions):
    return maps.to_grid(
            world_point[2],
            world_point[0],
            grid_dimensions,    
            pathfinder=sim.pathfinder,
        )

def to_world_coord(sim, map_point, grid_dimensions):
    new_pos_x, new_pos_y = maps.from_grid(
                    map_point[0],
                    map_point[1],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )

    return [new_pos_y, 0, new_pos_x]

def pyquaternion_to_quaternion(quat):
    return np.quaternion(quat.scalar, *quat.vector)

def distance_betwen_points(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2) ** 0.5