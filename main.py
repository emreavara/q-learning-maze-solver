import numpy as np
import pandas as pd

# map_information ={(1,1) : "W", (2,3) : "W", (0,5) : "W", (4,3) : "W",(2,7) : "W", (5,7) : "W", (5,6) : "W","information" : {"rows":6,"columns":8, "start":(2,1), "end":(0,7)}}
map_information = {(0,1) : "W", (3,1) : "W", (3,3) : "W", (3,4) : "W", "information" : {"rows":5,"columns":5, "start":(0,4), "end":(4,2)}}


# Initilizations

LEARNING_RATE = 0.5
GAMMA = 0.9
NUMBER_OF_ITERATIONS = 500

ROW_COUNT = map_information["information"]["rows"]
COLUMN_COUNT = map_information["information"]["columns"]
START_POINT_LOCATION = map_information["information"]["start"]
END_POINT_LOCATION = map_information["information"]["end"]
directions = {'Up':0,'Down':1,'Left':2,'Right':3}



def movement(direction,direction_row,direction_column):

    if direction == 0:
        direction_row -= 1
    elif direction == 1:
        direction_row += 1
    elif direction == 2:
        direction_column -= 1
    elif direction == 3:
        direction_column += 1
    return direction_row,direction_column

def find_location_of_walls(map_info):

    locations_of_walls = []
    for items in map_info.items():
        if items[1] == "W":
            locations_of_walls.append(items[0])

    return locations_of_walls





def rewarding(new_state_row,new_state_col):
    position = (new_state_row,new_state_col)
    if(position in location_of_walls):
        return -1000
    elif(position == END_POINT_LOCATION):
        return 100
    else:
        return 1

def find_path(start_location,end_location):
    path = []
    path.append(start_location)

    next_state = start_location
    while(next_state != end_location):
        row, column = next_state
        best_direction = np.argmax(Q_values[row, column, :])
        next_row, next_column = movement(best_direction,row,column)
        next_state = (next_row, next_column)
        path.append(next_state)

    print(path)


# Q Learning Algorithm

Q_values = np.zeros((ROW_COUNT,COLUMN_COUNT,len(directions))) # Initilization of Q values to zero
location_of_walls = find_location_of_walls(map_information)

for iteation in range(NUMBER_OF_ITERATIONS):
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            for action in directions.values():
                next_row,next_column = movement(action,row,col)
                if((next_row<0 or next_row>=ROW_COUNT) or (next_column<0 or next_column >= COLUMN_COUNT)):
                    Q_values[row,col,action] = -10000
                else:
                    # new_state = row*column_count + column
                    reward = rewarding(next_row,next_column)
                    #Q_values[state, action] = (1-learning_rate)*Q_values[state, action] + learning_rate * (reward + gamma * np.max(Q_values[new_state, :])) # Updated Q values
                    Q_values[row,col,action] = Q_values[row,col,action] + LEARNING_RATE * (reward + GAMMA * np.max(Q_values[next_row,next_column,:]) - Q_values[row,col,action])

find_path(START_POINT_LOCATION,END_POINT_LOCATION)

