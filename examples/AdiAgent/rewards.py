#########
# GOALS #
#########
import math


def goal_reached(g1, g2, state, next_state):
    # no goal in prev state, goal in next state
    # no goal in either state but next state is new level

    # Goal has now been reached since previous state
    # OR goal was not reached in previous state but either:
    # 1. level has changed meaning it would have been reached, OR
    # 2. The level is the same but has repeated, indicated by the num_repeats field incrementing
    # Repeating in this way does not apply to team losses and level resets
    return (not g1["reached"] and g2["reached"]) \
        or (not g1["reached"]
            and (state["content"]["gameData"]["level"] != next_state["content"]["gameData"]["level"]))


def check_new_level(state, next_state):
    return state["content"]["gameData"]["level"] != next_state["content"]["gameData"]["level"]
    # or \
    #    (self.prev_state["content"]["gameData"]["level"] == next_state["content"]["gameData"]["level"] and
    #     self.prev_state["content"]["gameData"]["num_repeats"] < next_state["content"]["gameData"]["num_repeats"])


###################
# ACTION PLANNING #
###################
import json


def has_moved(p1, p2):
    """
    Checks if player has moved since last state.
    :param p1: Player info from previous state.
    :param p2: Player info from current state.
    :return: True/False
    """
    return p1["x"] != p2["x"] or p1["y"] != p2["y"]


################
# PIN PLANNING #
################

def check_pin_placement(p1, p2, next_state):
    x = p2["pinCursorX"]
    y = p2["pinCursorY"]
    # No pin placed
    if x is None or y is None:
        return False

    for obj in next_state["content"]["scene"]:
        # Pin was placed on object
        if x == obj["x"] and y == obj["y"]:
            # This check avoids giving repeated awards for placing pin. The reward should only be given once
            # Check that the location of the pin cursor from one state to the next has changed
            if p1["pinCursorX"] != p2["pinCursorX"] and p1["pinCursorY"] != p2["pinCursorY"]:
                return True
    return False


##########
# COMBAT #
##########

def check_combat_outcome(self):
    """
    Checks the outcome of a combat event. Because combat is triggered while
    players move, the resulting state of the final action plan being submitted
    may include combat during player movement or enemy movement. Thus, this function
    will check the previous and next states and use the following criteria to determine
    combat outcome. Note, during this period it is possible the player has won and lost
    combat multiple times.
        1. Need to figure this out.
    :return:
    """
    pass


def health_lost_or_dead(p1, p2):
    """
    Checks difference between previous and next state to determine if a player has lost health or died
    :param next_state: The state resulting from the previous action
    :return: True if a player has lost health or died, False otherwise
    """
    return (p1["health"] < p2["health"]) or p2["dead"]  # or (not p1["dead"] and p2["dead"])


def enemy_reduced(state1, state2):
    """
    Checks if there are fewer monsters in the next state compared to previous state
    :param p1: The json file for the first scene
    :param p2: The json file for the second scene
    :return: True if there are fewer enemies now, false otherwise
    """
    # Checked!
    enemies1 = (helper_count_entity_in_scene(state1, "monster"))
    enemies2 = (helper_count_entity_in_scene(state2, "monster"))
    return enemies1 > enemies2

def closer_to_entity(state1, state2, entityFromName, entityType = "monster"):
    """
    Checks if the closest enemy to the entity with name "entityFromName" is closer in the next state
    :param state1: The json file for the first scene
    :param state2: The json file for the second scene
    :param entityFromName: The name of the entity we're trying to get everything from
    :param entityType: Optinal parameter for entity type we're trying to find
    :return: True if there is a closer enemy of given entityType
    """
    # Finds the entity from for both state1 and state2
    entityOne = find_entity_helper(state1, entityFromName)
    entityTwo = find_entity_helper(state2, entityFromName)
    # finds the closest entities distance in both states
    closestOne = closest_entity_helper(state1, entityOne, entityType)
    closestTwo = closest_entity_helper(state2, entityTwo, entityType)
    if(closestOne == float('inf') and closestTwo == float('inf')):
        return True
    return closestOne > closestTwo

def find_entity_helper(state,name):
    """
    Helper function to find an entity in the json file given the name
    :param state: the state
    :param name: name we're trying to find
    :return: The entire entity
    """
    for entity in state['content']['scene']:
        try:
            entity['entityType'].lower()
            if entity['entityType'].lower() == name.lower():
                return entity
                break
        except KeyError:
            pass
    return None

def closest_entity_helper(state, entityFrom, entityType="monster"):
    """
    Helper function to find the closest Entity from the entity in entityFrom
    :param state: the state we are looking through
    :param entityFrom: the entity from which we're looking
    :param entityType: the entity type we are trying to look at the closest for
    :return: -1 if entities is empty. Distance to closest entity otherwise.
    """
    entities = entity_list_helper(state, entityType)
    #returns negative one if there are no entities left to look at
    if(len(entities) == 0):
        return -1
    minimum = float('inf')
    x = entityFrom['x']
    y = entityFrom['y']
    #finds the minimum distance
    for entity in entities:
        distance = math.sqrt((x - entity['x']) ** 2 + (y - entity['y']) ** 2)
        minimum = min(distance, minimum)
    return minimum
def entity_list_helper(state, entityType= 'monster', category = 'entityType'):
    """
    Helper function to provide a list of all the entities of a certain type
    :param state: the state we're looking for
    :param entityType: optional parameter to specify the entity type we are looking
    :param category: optional parameter for category we're looking for
    :return: a list of all the entities of a certain type
    """
    entityList = []
    for obj in state['content']['scene']:
        if entityType.lower() in obj[category].lower():
            entityList.append(obj)
    return entityList


def helper_calculate_distance(json_data, entityfrom, entityto, fromCat='objectCode', toCat='objectCode'):
    """
    Helper method to calculate distance between two entities given a JSON file. Will only work for UNIQUE KEYS;
    :param json_data: The JSON data for the scene
    :param entityfrom: The name of the first entity
    :param entityto: The name of the second entity
    :param fromCat: The category for the 'from' entity name (default is 'name')
    :param toCat: The category for the 'to' entity name (default is 'name')
    :return: The distance between the two entities
    """
    entity1 = None
    entity2 = None
    #calculate's the distance between two entities if we have not found either one
    for entity in json_data['content']['scene']:
        try:
            if entity[fromCat].lower() == entityfrom.lower():
                entity1 = (entity['x'], entity['y'])
            elif entity[toCat].lower() == entityto.lower():
                entity2 = (entity['x'], entity['y'])
            if entity1 and entity2:
                break
        except KeyError:
            # Handle the case where 'name' key doesn't exist in the entity dictionary
            pass

    if entity1 is None or entity2 is None:
        return None

    # Calculate distance
    distance = math.sqrt((entity2[0] - entity1[0]) ** 2 + (entity2[1] - entity1[1]) ** 2)

    return distance


def helper_count_entity_in_scene(json_data, entity):
    """
    Helper method to count the number of an entity in the scene
    :param json_data: The json file for the first scene
    :param entity: A tag inside the entity type that we want to count
    :return: integer length of entity count
    """
    # counts all the entity of a certain type
    count = sum(1 for obj in json_data['content']['scene'] if entity.lower() in obj['entityType'].lower())
    return count

"""
Code to help with testing
:with open("/Users/adikrish/PycharmProjects/TestingJson/example_unity_state.json", "r") as file1:
    state1 = json.load(file1)

with open("/Users/adikrish/PycharmProjects/TestingJson/example_unity_statecopy.json", "r") as file2:
    state2 = json.load(file2)
print(closer_to_entity(state1, state2, "Human"))
"""
