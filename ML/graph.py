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


def generate_next_states(current_state, nodes, agent, goal_node):
    next_states = []
    limbs = {"right_hand", "left_hand", "right_foot", "left_foot"}

    # Parameters for reachability
    max_vertical_reach = agent.vertical_reach  # Feet to hands max distance
    max_horizontal_reach = agent.horizontal_reach  # Hand-to-hand max horizontal distance

    # Get current positions
    hand_positions = {
        "right_hand": next(node for node in nodes if node['id'] == current_state.state["right_hand"]),
        "left_hand": next(node for node in nodes if node['id'] == current_state.state["left_hand"])
    }
    foot_positions = {
        "right_foot": next(node for node in nodes if node['id'] == current_state.state["right_foot"]),
        "left_foot": next(node for node in nodes if node['id'] == current_state.state["left_foot"])
    }

    for limb in limbs:
        current_node_id = current_state.state[limb]

        # Generate potential moves for this limb
        for target_node in nodes:
            if target_node['id'] == current_node_id:
                continue  # Skip if the limb is already on this node

            # Enforce constraints based on limb type
            if "hand" in limb:

                # No crossing your arms >:(
                other_hand = "left_hand" if limb == "right_hand" else "right_hand"
                if other_hand == "left_hand": # The stationary one
                    if target_node['center'][0] < hand_positions[other_hand]['center'][0]:
                        continue
                else:  # other_hand == "right_hand"
                    if target_node['center'][0] > hand_positions[other_hand]['center'][0]:
                        continue

                # Check vertical reach (feet to hands)
                lowest_foot = foot_positions["right_foot"] if foot_positions["right_foot"]['center'][1] < foot_positions["left_foot"]['center'][1] else foot_positions["left_foot"]
                if abs(euclidean_distance(target_node['center'], lowest_foot['center'])) > max_vertical_reach:
                    continue

                # Check horizontal reach (hand-to-hand)
                other_hand_pos = hand_positions[other_hand]['center']
                if abs(euclidean_distance(target_node['center'], other_hand_pos)) > max_horizontal_reach:
                    continue

            elif "foot" in limb:
                # foot can't go above lowest hand
                lowest_hand = hand_positions["right_hand"] if hand_positions["right_hand"]['center'][1] < hand_positions["left_hand"]['center'][1] else hand_positions["left_hand"]
                if target_node['center'][1] <= lowest_hand['center'][1]:
                    continue

                # Check feet proximity (feet should not be too far apart)
                other_foot = "left_foot" if limb == "right_foot" else "right_foot"
                other_foot_pos = foot_positions[other_foot]['center']
                if abs(euclidean_distance(target_node['center'], other_foot_pos)) > (max_vertical_reach / 2):  # divide by 2 to shorten the distance the leg can go
                    continue

            # Add valid move
            limb_positions = deepcopy(current_state.state)
            limb_positions[limb] = target_node['id']
            new_state = state.Node(F=0, g=current_state.g + 1, h=get_heuristic(nodes, goal_node, limb_positions), state=limb_positions, parent=deepcopy(current_state.parent))
            new_state.parent.append(current_state.state)
            next_states.append(new_state)
    return next_states


def a_star(nodes, agent, start_state, goal_node_id):
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
        for next_state in generate_next_states(current_state, nodes, agent, goal_node):
            if tuple(next_state.state.values()) in explored:
                continue
            heappush(frontier, next_state)

    return None  # No path found

# Example Usage
def find_path(nodes, agent):
    start_state = {
        "right_hand": 2,  # Starting node for right hand
        "left_hand": 2,  # Starting node for left hand
        "right_foot": 1,  # Starting node for right foot
        "left_foot": 1  # Starting node for left foot
    }
    goal_node_id = len(nodes)  # Goal node ID

    steps = a_star(nodes, agent, start_state, goal_node_id)

    print(steps)
    print(len(steps))

    for step in steps:
        print(step)

    # Output the steps
    '''
    if steps:
        for i, step in enumerate(steps):
            print(f"Step {i + 1}: Move {step['limb']} to node {step['node_id']}")
    else:
        print("No path found.")
    '''
