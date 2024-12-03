import math
import state
from heapq import heappush, heappop
from copy import deepcopy


def euclidean_distance(pos1, pos2):
    x1, y1 = tuple(pos1)
    x2, y2 = tuple(pos2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def is_within_reach(node1, node2, reach_radius):
    return euclidean_distance(node1, node2) <= reach_radius

def get_state_center(nodes, ids: tuple):
    positions = []
    for id in ids:
        for node in nodes:
            if node['id'] == id:
                positions.append(tuple(node['center']))

    x, y = 0, 0
    for pos_x, pos_y in positions:
        x += pos_x
        y += pos_y

    x /= len(positions)
    y /= len(positions)

    return x, y

def get_heuristic(nodes, goal_node, start_state):
    return euclidean_distance(tuple(get_state_center(nodes, tuple(start_state.values()))), goal_node['center'])

def reconstruct_path(node):
    node.parent.append(node.state)
    return node.parent


def print_moves(steps: list):  # Expecting a list of dictionaries
    step_string = []
    step_string.append(f"Starting State: {steps[0]}")
    
    for x in range(len(steps) - 1):
        curr_state = steps[x]
        next_state = steps[x + 1]
        for limb in next_state:
            if curr_state[limb] == next_state[limb]:
                continue
            new_value = next_state[limb]
            step_string.append(f"{limb} move to hold number {new_value}")
    return step_string


def generate_next_states(current_state, nodes, agent, goal_node, foot_hold_ids):
    next_states = []
    limbs = ["right_hand", "left_hand", "right_foot", "left_foot"]

    # Parameters for reachability
    max_vertical_reach = agent.vertical_reach  # Feet to hands max distance
    max_horizontal_reach = agent.horizontal_reach * 0.8  # Hand-to-hand max horizontal distance

    # Get current positions
    hand_positions = {
        "right_hand": next(node for node in nodes if node['id'] == current_state.state["right_hand"]),
        "left_hand": next(node for node in nodes if node['id'] == current_state.state["left_hand"])
    }
    foot_positions = {
        "right_foot": next(node for node in nodes if node['id'] == current_state.state["right_foot"]),
        "left_foot": next(node for node in nodes if node['id'] == current_state.state["left_foot"])
    }

    # Determine the lowest hand position (largest y if y increases downward)
    lowest_hand_y = max(hand_positions["right_hand"]['center'][1],
                       hand_positions["left_hand"]['center'][1])

    for limb in limbs:
        current_node_id = current_state.state[limb]

        # Generate potential moves for this limb
        for target_node in nodes:
            if target_node['id'] == current_node_id:
                continue  # Skip if the limb is already on this node

            # Enforce constraints based on limb type
            if "hand" in limb:

                # If the target hold is a foot hold, skip it
                if target_node['id'] in foot_hold_ids:
                    continue

                # Prevent crossing arms
                other_hand = "left_hand" if limb == "right_hand" else "right_hand"
                if limb == "right_hand":
                    # Right hand should be to the right of left hand
                    if target_node['center'][0] < hand_positions[other_hand]['center'][0]:
                        continue
                else:
                    # Left hand should be to the left of right hand
                    if target_node['center'][0] > hand_positions[other_hand]['center'][0]:
                        continue

                # Check vertical reach (from feet to hands)
                lowest_foot = foot_positions["right_foot"] if foot_positions["right_foot"]['center'][1] < foot_positions["left_foot"]['center'][1] else foot_positions["left_foot"]
                if euclidean_distance(target_node['center'], lowest_foot['center']) > max_vertical_reach:
                    continue

                # Check horizontal reach (hand-to-hand)
                other_hand_pos = hand_positions[other_hand]['center']
                if euclidean_distance(target_node['center'], other_hand_pos) > max_horizontal_reach:
                    continue

                # Assign lower movement cost for hands
                move_cost = 1

            elif "foot" in limb:

                other_foot = "left_foot" if limb == "right_foot" else "right_foot"

                # If the target hold is a foot hold and the other foot is already on it, skip
                if target_node['id'] in foot_hold_ids and foot_positions[other_foot]['id'] == target_node['id']:
                    continue

                # Prevent feet from occupying the same hold as hands
                hands = [hand_positions["right_hand"]['id'], hand_positions["left_hand"]['id']]
                if target_node['id'] in hands:
                    continue

                # Prevent criss-crossing feet
                if limb == "right_foot":
                    # Right foot should be to the right of left foot
                    if target_node['center'][0] < foot_positions["left_foot"]['center'][0]:
                        continue
                else:
                    # Left foot should be to the left of right foot
                    if target_node['center'][0] > foot_positions["right_foot"]['center'][0]:
                        continue

                # Calculate the maximum allowed y-coordinate for the foot
                max_foot_y = lowest_hand_y + 0.4 * agent.height

                if target_node['center'][1] < max_foot_y:
                    continue

                # Check feet proximity (feet should not be too far apart)
                other_foot_pos = foot_positions[other_foot]['center']
                if euclidean_distance(target_node['center'], other_foot_pos) > (max_vertical_reach / 2):
                    continue

                move_cost = 3

            else:
                continue 

            # Add valid move
            limb_positions = deepcopy(current_state.state)
            limb_positions[limb] = target_node['id']
            new_state = None

            # Define heuristic based on limb type
            if "hand" in limb:
                h = euclidean_distance(target_node['center'], goal_node['center'])
            elif "foot" in limb:
                h = min(
                    euclidean_distance(target_node['center'], hand_positions["right_hand"]['center']),
                    euclidean_distance(target_node['center'], hand_positions["left_hand"]['center'])
                )

            # Calculate g(n) and f(n)
            g = current_state.g + move_cost
            f = g + h

            new_state = state.Node(
                F=f,
                g=g,
                h=h,
                state=limb_positions,
                parent=deepcopy(current_state.parent)
            )
            new_state.parent.append(current_state.state)
            next_states.append(new_state)

    return next_states





def a_star(nodes, agent, start_state, goal_node_id, foot_hold_ids):
    goal_node = next(node for node in nodes if node['id'] == goal_node_id)

    # Priority queue for A*
    frontier = []
    heappush(frontier, state.Node(F=0, g=0, h=euclidean_distance(tuple(get_state_center(nodes, tuple(start_state.values()))), goal_node['center']), state=start_state, parent=[]))  # (priority, state, path)
    explored = set()

    while frontier:
        current_state = heappop(frontier)

        # Check if either hand reaches the goal node
        if current_state.state["right_hand"] == goal_node_id or current_state.state["left_hand"] == goal_node_id:
            path_taken = reconstruct_path(current_state)
            return path_taken  # Goal reached

        # Avoid revisiting states
        state_tuple = tuple(current_state.state.values())
        explored.add(state_tuple)

        # Generate next states
        for next_state in generate_next_states(current_state, nodes, agent, goal_node, foot_hold_ids):
            if tuple(next_state.state.values()) in explored:
                continue
            heappush(frontier, next_state)

    return None  # No path found

# Example Usage
def find_path(nodes, agent, foot_hold_ids, start_state):
    goal_node_id = len(nodes)  # Goal node ID

    steps = a_star(nodes, agent, start_state, goal_node_id, foot_hold_ids)


    # Output the steps
    return print_moves(steps)
    '''
    if steps:
        for i, step in enumerate(steps):
            print(f"Step {i + 1}: Move {step['limb']} to node {step['node_id']}")
    else:
        print("No path found.")
    '''
