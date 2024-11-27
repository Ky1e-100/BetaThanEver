# detect_hold w img path, then filter

import inference
import agent
import graph

# Gather the holds
test_img = '../dataset/test/test2.jpg'

holds = inference.detect_holds(test_img)
filtered_holds_by_y = inference.filter_holds(holds, "Pink")
filtered_holds_by_x = sorted(filtered_holds_by_y, key=lambda x: x['center'][0], reverse=True)

# re-id the holds
for x in range(len(filtered_holds_by_y)):
    filtered_holds_by_y[x]['id'] = x + 1

# Determine the height/width of the puzzle
puzzle_height = filtered_holds_by_y[0]['center'][1]
puzzle_width = filtered_holds_by_x[0]['center'][0]

# Get the users height
print("Please input your height (in centimeters)")
user_height = int(input())

# Determine the users 'scaled' height to be in ratio with yolov5's units
avg_wall_height = 500  # in cm, for ratios
scaled_user_height = (user_height / avg_wall_height) * puzzle_height

# Create the agent
climber = agent.climber(scaled_user_height)

graph.find_path(filtered_holds_by_y, climber)


