# detect_hold w img path, then filter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../ML'))
import inference
import agent
import graph

def path(holds, user_height, foot_holds, colour, start_state):
    # Gather the holds

    filtered_holds_by_y = inference.filter_holds(holds, colour)
    filtered_holds_by_x = sorted(filtered_holds_by_y, key=lambda x: x['center'][0], reverse=True)

    # re-id the holds
    for x in range(len(filtered_holds_by_y)):
        filtered_holds_by_y[x]['id'] = x + 1

    # Determine the height/width of the puzzle
    puzzle_height = filtered_holds_by_y[0]['center'][1]
    puzzle_width = filtered_holds_by_x[0]['center'][0]


    foot_hold_ids_str = foot_holds.split()
    foot_hold_ids = []
    for x in foot_hold_ids_str:
        foot_hold_ids.append(int(x))


    # Determine the users 'scaled' height to be in ratio with yolov5's units
    avg_wall_height = 500  # in cm, for ratios
    scaled_user_height = (int(user_height) / avg_wall_height) * puzzle_height

    # Create the agent
    climber = agent.climber(scaled_user_height)

    return graph.find_path(filtered_holds_by_y, climber, foot_hold_ids, start_state)


