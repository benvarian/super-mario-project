''' CURRENT BEST Rule based Mario agent
- testing visualisation techniques'''
# Import packages
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import matplotlib.pyplot as plt
# Import Lauren Gee's location_objects function 
from lauren_gee_locate_objects import locate_objects
from visualisation_graphs import * 

# Declare global variables ##############################################################
# Store mario's last 20 world location coordinates to see how he is moving
MARIO_WORLD_LOCATION_REC = [(0,0)] * 20
# Margin error to account for possible computer vision error in x and y coordinates returned
MARGIN_RAND_ERR = 3
# Margin for looking ahead on path - how far in front should mario 'react' to holes
MARGIN_PATH_AHEAD = 6
# Margin for looking ahead at obstacles - how far in front should mario 'react' to enemies and blocks/pipes
MARGIN_OBS_AHEAD = 10

def make_action(screen, info, step, env, prev_action):
    
 
    def calculate_object_position(location_data):
        '''
        Function takes an object location record and
        returns 4 corner coordinates - left x, right x, top y, bottom y
        - locate_objects() method only returns top left x-y coordinates
        - used to identify where object starts and ends on screen
        '''
        location, dimensions, object_name = location_data # unpack location data
        object_width, object_height = dimensions # get width and height
        
        object_x_left, object_y_top = location 
        object_x_right = object_x_left + object_width
        object_y_bottom = object_y_top + object_height # add height as y axes is 0 at top of screen
        
        return (object_x_left, object_x_right), (object_y_top, object_y_bottom)
    
    def extract_relevant_blocks(obstacles_locations, mario_location):
        '''
        Function takes a record of mario and a list of obstacle records
        and effectively splits the locations into ones which are above
        mario's feet and ones which are below
        - This is to filter obstacles the agent could run into
        '''
        above_obstacles = []
        below_obstacles = []
        
        x_coor, y_coor = calculate_object_position(mario_location)
        mario_y_bottom = y_coor[1] # extract mario's bottom y coordinate - i.e., his feet
        
        for obs in obstacles_locations:
            if obs[0][1] >= mario_y_bottom - MARGIN_RAND_ERR: # if top of obstacle is below mario's feet
                below_obstacles.append(obs)
            else:
                above_obstacles.append(obs)

        return above_obstacles, below_obstacles

    def is_obstacle_ahead(mario_data, obs_data):
        '''
        Function takes mario data record and a obstacle data record and
        returns true if any part of the block is directly in front of mario
        - used to indicate if mario should jump as a next action
        '''
        # get mario position coordinates
        x_coors, y_coors = calculate_object_position(mario_data)
        mario_x_left, mario_x_right = x_coors
        mario_y_top, mario_y_bottom = y_coors
        
        # get obstacle data
        x_coors, y_coors = calculate_object_position(obs_data)
        obs_x_left, obs_x_right = x_coors
        obs_y_top, obs_y_bottom = y_coors
        
        return (
                # is the object in mario's line of sight (y coordinates)
                ( not (mario_y_bottom < obs_y_top or mario_y_top> obs_y_bottom)) and
                # is the object in front of mario
                (-MARGIN_RAND_ERR < obs_x_left - mario_x_right < MARGIN_OBS_AHEAD) )
    
    def is_safe_ahead(mario_data, block_data):
        '''
        Function takes mario data record and an obstacle data record and
        returns True if mario can safely walk right without falling due to block
        '''
        # get mario position coordinates
        x_coors, y_coors = calculate_object_position(mario_data)
        mario_x_left, mario_x_right = x_coors
        mario_y_top, mario_y_bottom = y_coors
        
        # get obstacle data
        x_coors, y_coors = calculate_object_position(block_data)
        obs_x_left, obs_x_right = x_coors
        obs_y_top, obs_y_bottom = y_coors
        
        return (
                # this block is directly below mario:
                ( (obs_y_top - mario_y_bottom < MARGIN_RAND_ERR) and
                 (obs_x_left < mario_x_right < obs_x_right - MARGIN_RAND_ERR) )
                or 
                # this block will be directly below mario if he moves right:
                ( (obs_y_top - mario_y_bottom < MARGIN_RAND_ERR) and 
                 (obs_x_left < mario_x_right + MARGIN_PATH_AHEAD < obs_x_right)) )
        
    def best_action(object_locations):
        '''
        Function takes object_locations and returns
        the best move
        '''
        # get mario's location:
        mario_location = object_locations["mario"][0]
        # list of locations of enemies, such as goombas and koopas:
        enemy_locations = object_locations["enemy"]
        # list of locations of blocks, pipes, etc:
        block_locations = object_locations["block"]
        # split blocks as being above mario's feet and below mario's feet to reduce processing time
        above_block_locations, below_block_locations = extract_relevant_blocks(block_locations, mario_location)
        
        # First, check if mario should jump to avoid an enemy or obstacle:
        # iterate over all blocks above mario's feet and enemies in frame
        for obs in enemy_locations + above_block_locations:
            # if the obstacle is in front of mario and blocking his path
            if is_obstacle_ahead(mario_location, obs):
                # if mario is descending (i.e., he jumped previously and his y coordinates has been decreasing)
                if MARIO_WORLD_LOCATION_REC[-1][1] < MARIO_WORLD_LOCATION_REC[-2][1]:
                    # this is important so that mario can jump again as soon as he lands on the ground
                    return 1 # return 'Right' action
            
                # if an obstacle is ahead and mario is not descending, he should jump
                else:
                    return 2 # return 'Right + A' action
        
        # Second, check if mario should jump to avoid a cavity in ground
        safe_ahead = False
        for block in below_block_locations:
            if is_safe_ahead(mario_location, block):
                safe_ahead = True
                
        if safe_ahead:
            # if safe path ahead, return 'Right' action
            return 1
        else:
            # otherwise it is not safe; mario should jump
            return 2 # return 'Right + A' action
        
    # Extract object location records and find best move to make
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)
    return best_action(object_locations)

##################################### PLAY GAME #########################################
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
time_vs_x_line = Time_vs_X_line()
time_vs_x_scat = Time_vs_X_scatter()

for step in range(100000):
    # try to find the best action
    try:
        action = make_action(obs, info, step, env, action)
    # Failsafe: if something fails, choose a random action
    except:
        action = env.action_space.sample()

    # complete chosen action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Failsafe: if mario has been stuck in the same position
    if(MARIO_WORLD_LOCATION_REC.count(MARIO_WORLD_LOCATION_REC[0]) == len(MARIO_WORLD_LOCATION_REC)):
        # reset controls by actioning 'Right'
       obs, reward, terminated, truncated, info = env.step(1)
    
    # update mario's world position
    MARIO_WORLD_LOCATION_REC.pop(0)
    MARIO_WORLD_LOCATION_REC.append((info["x_pos"], info["y_pos"]))
    
    # add plot points
    time_vs_x_line.add_record(x_instance=info["x_pos"], time_instance=400-info["time"])
    time_vs_x_scat.add_record(x_instance=info["x_pos"], time_instance=400-info["time"], action=action)
    
    done = terminated or truncated
    if done or info["flag_get"]:
        time_vs_x_line.plot_graph()
        time_vs_x_scat.plot_graph()
        break
        env.reset()
env.close()
