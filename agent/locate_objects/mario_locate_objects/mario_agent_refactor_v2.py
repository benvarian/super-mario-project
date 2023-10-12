''' Rule based Mario agent'''
# Import packages
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
# Import Lauren Gee's location_objects function 
from lauren_gee_locate_objects import locate_objects

MARGIN_ERROR = 3
MARGIN_PATH_AHEAD = 5

LAST_WORLD_X_Y = [(0,0)] * 20

def make_action(screen, info, step, env, prev_action):
    
 
    def calculate_object_position(location_data):
        '''
        Function returns 4 corner coordinates - left x, right x, top y, bottom y
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
            if obs[0][1] >= mario_y_bottom - MARGIN_ERROR: # if top of obstacle is below mario's feet
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
                ( not (mario_y_bottom + 50< obs_y_top or mario_y_top> obs_y_bottom)) and
                # is the object in front of mario
                (-8 < obs_x_left - (mario_x_right) < 10) )
    
    def is_safe_ahead(mario_data, obs_data):
        '''
        Function takes mario data record and an obstacle data record and
        returns True if mario can safely walk right without falling
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
                # this block is directly below mario:
                ( (obs_y_top - mario_y_bottom < MARGIN_ERROR) and
                 (obs_x_left < mario_x_right < obs_x_right - MARGIN_ERROR) )
                or 
                # this block will be directly below mario if he moves right:
                ( (obs_y_top - mario_y_bottom < MARGIN_ERROR) and 
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
                if LAST_WORLD_X_Y[-1][1] < LAST_WORLD_X_Y[-2][1]:
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

   
################################################################################
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    if(LAST_WORLD_X_Y.count(LAST_WORLD_X_Y[0]) == len(LAST_WORLD_X_Y)):
        obs, reward, terminated, truncated, info = env.step(1)
    LAST_WORLD_X_Y.pop(0)
    LAST_WORLD_X_Y.append((info["x_pos"], info["y_pos"]))
    
    done = terminated or truncated
    if done:
        env.close()
        #env.reset()
env.close()
