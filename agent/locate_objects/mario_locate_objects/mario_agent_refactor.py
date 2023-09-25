# PASSES FIRST LOT OF PIPES

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = False
PRINT_LOCATIONS = False

MARGIN_ERROR = 3
MARGIN_PATH_AHEAD = 5

LAST_WORLD_X_Y = [(0,0)] * 20
LAST_MOVES = [None] * 20

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
    print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

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
    print(LAST_WORLD_X_Y)
    if(LAST_WORLD_X_Y.count(LAST_WORLD_X_Y[0]) == len(LAST_WORLD_X_Y)):
        obs, reward, terminated, truncated, info = env.step(1)
    LAST_WORLD_X_Y.pop(0)
    LAST_WORLD_X_Y.append((info["x_pos"], info["y_pos"]))
    
    done = terminated or truncated
    if done:
        print("GAME OVER")
        env.close()
        #env.reset()
env.close()
