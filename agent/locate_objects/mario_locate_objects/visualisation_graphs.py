import matplotlib.pyplot as plt
import numpy as np

class Time_vs_X_line:
    def __init__(self):
        self.mario_x = []
        self.time_taken = []
        
    def add_record(self, x_instance, time_instance):
        self.mario_x.append(x_instance)
        self.time_taken.append(time_instance)
    
    def plot_graph(self):
        plt.plot(self.time_taken, self.mario_x)
        plt.xlabel('Time taken (s)')
        plt.ylabel('Progress (world x)')
        plt.title('Mario world progress over time')
        plt.show()
        
class Time_vs_X_scatter:
    def __init__(self):
        self.mario_x_1 = []
        self.time_taken_1 = []
        self.mario_x_2 = []
        self.time_taken_2 = []
        
    def add_record(self, x_instance, time_instance, action):
        if action == 1:
            self.mario_x_1.append(x_instance)
            self.time_taken_1.append(time_instance)
        elif action == 2:
            self.mario_x_2.append(x_instance)
            self.time_taken_2.append(time_instance)
    
    def plot_graph(self):
        x = np.array(self.mario_x_1)
        y = np.array(self.time_taken_1)
        plt.scatter(x, y)
        
        x = np.array(self.mario_x_2)
        y = np.array(self.time_taken_2)
        plt.scatter(x, y)
        
        plt.xlabel('Time taken (s)')
        plt.ylabel('Progress (world x)')
        plt.title('Mario world progress over time')
        plt.show()
        
        