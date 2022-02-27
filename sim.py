import habitat_sim
import numpy as np
import random

from config import *

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Camera configurations

    CAMERA_CONFIGURATIONS = [
        {
            "position": [0., settings['sensor_height'], 0.],
            "orientation": [0, 0, 0],
            "name": "front",
        },
#        {
#            "position": [0., settings['sensor_height'], 0.],
#            "orientation": [0, -1.57, 0],
#            "name": "right",
#        },
#        {
#            "position": [0., settings['sensor_height'], 0.],
#            "orientation": [0, 1.57, 0],
#            "name": "left",
#        },
#        {
#            "position": [0., settings['sensor_height'], 0.],
#            "orientation": [0, np.pi, 0],
#            "name": "back",
#        },
        # {
        #     "position": [],
        #     "orientation": [],
        #     "name": "",
        # },       
    ]

    sensors = {}

    for camera_config in CAMERA_CONFIGURATIONS:

        if settings['rgb']:
            sensors[f"{camera_config['name']}_RGB"] = {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": camera_config['position'],
                "orientation": camera_config['orientation'],
            }

        if settings['depth']:
            sensors[f"{camera_config['name']}_DEPTH"] = {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": camera_config['position'],
                "orientation": camera_config['orientation'],
            }

        if settings['semantic']:
            sensors[f"{camera_config['name']}_SEMANTIC"] = {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": camera_config['position'],
                "orientation": camera_config['orientation'],
            }

    # Note: all sensors must have the same resolution

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        # sensor_spec.hfov = mn.Deg(settings["fov"]

        sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_sim(BASE_DIR, scene="17DRP5sb8fy"):

    scene = f"{BASE_DIR}/{scene}.glb"

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = False  # @param {type:"boolean"}

    sim_settings = {
        "width": WIDTH,  # Spatial resolution of the observations
        "height": HEIGHT,
        "scene": scene,  # Scene path
        "default_agent": 0,
        "sensor_height": AGENT_HEIGHT,  # Height of sensors in meters
        "rgb": rgb_sensor,  # RGB sensor
        "depth": depth_sensor,  # Depth sensor
        "semantic": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
        "fov": str(HFOV),
    }

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    seed = 0
    random.seed(seed)
    sim.seed(seed)
    np.random.seed(seed)
   
    return sim
