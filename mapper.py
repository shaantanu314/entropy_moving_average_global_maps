import matplotlib.pyplot as plt
import cv2
import math

from habitat.utils.visualizations import maps
from PIL import Image

from config import COORDINATE_MAX, COORDINATE_MIN, RESOLUTION, NUM_SAMPLES
from geometry_utils import *

def make_global_map(sim, scale=0.05, height=0):
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=scale
    )

    return top_down_map

def recolor_map(map, new_colors=[1, 0, 0]):
    recolor_map_fn = np.array(
        new_colors
    )
    new_map = recolor_map_fn[map]
    return new_map

def get_mesh_occupancy(sim, agent_state, map_size=128, padding_size=600):

    top_down_map = make_global_map(sim)
    top_down_map = np.uint8(recolor_map(top_down_map))

    agent_position = agent_state.position
    agent_rotation = agent_state.rotation

    a_y, a_x = maps.to_grid(
        agent_position[2],
        agent_position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        pathfinder=sim.pathfinder
    )

    top_down_map_pad = np.pad(top_down_map, (padding_size, padding_size), mode="constant", constant_values=0)
    a_x += padding_size
    a_y += padding_size

    # Crop region centered around the agent
    mrange = int(map_size * 1.5)
    ego_map = top_down_map_pad[
            (a_y - mrange) : (a_y + mrange), (a_x - mrange) : (a_x + mrange)
        ]

    if ego_map.shape[0] == 0 or ego_map.shape[1] == 0:
        print("EMPTY")
        ego_map = np.zeros((2 * mrange + 1, 2 * mrange + 1), dtype=np.uint8)

    # Rotate to get egocentric map
    # Negative since the value returned is clockwise rotation about Y,
    # but we need anti-clockwise rotation
    agent_heading = compute_heading_from_quaternion(agent_rotation)
    agent_heading = math.degrees(agent_heading)

    half_size = ego_map.shape[0] // 2
    center = (half_size, half_size)
    M = cv2.getRotationMatrix2D(center, agent_heading, scale=1.0)

    # print(center, agent_heading)

    ego_map = (
        cv2.warpAffine(
            ego_map * 255,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1,),
        ).astype(np.float32)
        / 255.0
    )

    mrange = int(map_size)
    ego_map = ego_map[
        (half_size - mrange) : (half_size + mrange),
        (half_size - mrange) : (half_size + mrange),
    ]

    ego_map[ego_map > 0.5] = 1.0
    ego_map[ego_map <= 0.5] = 0.0

    # # This map is currently 0 if occupied and 1 if unoccupied. Flip it.
    # ego_map = 1.0 - ego_map

    # # Flip the x axis because to_grid() flips the conventions
    # ego_map = np.flip(ego_map, axis=1)

    # Get forward region infront of the agent
    half_size = ego_map.shape[0] // 2
    quarter_size = ego_map.shape[0] // 4
    center = (half_size, half_size)

    ego_map = ego_map[0:half_size, quarter_size : (quarter_size + half_size)]

    return ego_map

