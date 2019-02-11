import random
import matplotlib.pyplot as plt
import numpy as np

class Field:
    def __init__(self, w_length=3, w_width=2, t_length=2, t_width=1):
        self.w_length = 3
        self.w_width = 2
        self.t_length = t_length
        self.t_width = t_width
        
    def left_line(self, x):
        return (self.w_width - self.t_width)/2
    
    def right_line(self, x):
        return self.left_line(x) + self.t_width
    
    def left_line_points(self, grid_step=0.01):
        x_points = np.arange((self.w_length - self.t_length)/2, self.w_length-(self.w_length - self.t_length)/2, grid_step)
        point_vec = np.vectorize(self.left_line)
        return x_points, point_vec(x_points)
    
    def right_line_points(self, grid_step=0.01):
        x_points = np.arange((self.w_length - self.t_length)/2, self.w_length-(self.w_length - self.t_length)/2, grid_step)
        point_vec = np.vectorize(self.right_line)
        return x_points, point_vec(x_points)
    
    def low_line(self, y):
        return (self.w_length - self.t_length)/2
    
    def hight_line(self, y):
        return self.low_line(y) + self.t_length
    
    def low_line_points(self, grid_step=0.01):
        y_points = np.arange(0, self.w_length, grid_step)
        point_vec = np.vectorize(self.low_line)
        return point_vec(y_points), y_points

    def hight_line_points(self, grid_step=0.01):
        y_points = np.arange(0, self.w_length, grid_step)
        point_vec = np.vectorize(self.hight_line)
        return point_vec(y_points), y_points
    

    def show_field(self, robot, step, p, pr, weights, folder_name = "test"):
        plt.figure("Robot in the world", figsize=(5.*self.w_width, 5.*self.w_length))
        plt.title('Field. Step = ' + str(step))
        grid = [0, self.w_width, 0, self.w_length]
        plt.axis(grid)
        plt.grid(b=True, which='major', color='0.75', linestyle='--')
        
        
        data = self.left_line_points()
        plt.plot(data[1], data[0])
        
        
        data = self.right_line_points()
        plt.plot(data[1], data[0])
       
        
        data = self.low_line_points()
        plt.plot(data[1], data[0])
        
        
        data = self.hight_line_points()
        plt.plot(data[1], data[0])
        
        
        
        for ind in range(len(p)):
            # particle
            circle = plt.Circle((p[ind].x, p[ind].y), 0.05, facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
            plt.gca().add_patch(circle)
            '''
            # particle's orientation
            arrow = plt.Arrow(p[ind].x, p[ind].y, 2*np.cos(p[ind].orientation), 2*np.sin(p[ind].orientation), width=0.05, alpha=1., facecolor='#994c00', edgecolor='#994c00')
            plt.gca().add_patch(arrow)
             '''   
         # draw resampled particles
        for ind in range(len(pr)):
            
            # particle
            circle = plt.Circle((pr[ind].x, pr[ind].y), 0.05, facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
            plt.gca().add_patch(circle)
            '''
            # particle's orientation
            arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2*np.cos(pr[ind].orientation)/20, 2*np.sin(pr[ind].orientation)/20, alpha=1., facecolor='#006600', edgecolor='#006600')
            plt.gca().add_patch(arrow)
            '''
        # robot's location
        circle = plt.Circle((robot.x, robot.y), 0.05, facecolor='#6666ff', edgecolor='#0000cc')
        plt.gca().add_patch(circle)

        # robot's orientation
        arrow = plt.Arrow(robot.x, robot.y, 2*np.cos(robot.orientation)/20, 2*np.sin(robot.orientation)/20, width=0.05, alpha=0.5, facecolor='#000000', edgecolor='#000000')
        plt.gca().add_patch(arrow)

        plt.savefig(folder_name + "figure_" + str(step) + ".png")
        plt.close()
        
        
class Robot(Field):
 
    def __init__(self, x = 1, y = 0.5):
 
        self.x = x          # robot's x coordinate
        self.y = y          # robot's y coordinate
        self.orientation = np.pi/2   # robot's orientation
 
        self.forward_noise = 0.05   # noise of the forward movement
        self.turn_noise = 0.1      # noise of the turn
        self.sense_noise = 0.05     # noise of the sensing
        
    def set(self, new_x, new_y, new_orientation):
        #if new_orientation < 0 or new_orientation >= 2 * pi:
        #   raise ValueError('Orientation must be in [0..2pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
 
        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sense_noise)  

    def sense(self):
        z = []

        for i in range(len(landmarks)):
            dist =np. sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            z.append(dist)

        return z
    def move(self, turn, forward):
 

        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * np.pi

        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (np.cos(orientation) * dist)
        y = self.y + (np.sin(orientation) * dist)

        # cyclic truncate
        x %= Field().w_width
        y %= Field().w_length

        # set particle
        res = Robot()
        res.set(x, y, orientation)
        #res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

        return res
    
    def gaussian(self, mu, sigma, x):
        #print(mu, sigma, x)
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))
    
    def measurement_prob(self, measurement):
        prob = 1.0
        for i in range(len(landmarks)):
            dist = np.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
class ParticleFilter():
    def __init__(self, myrobort, 
                 n = 1000, forward_noise = 0.025, 
                 turn_noise = 0.1, sense_noise = 0.05):
        
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.n = n  # number of particles
        self.myrobot = myrobot
        self.p = [] 
        
        for i in range(self.n):
            
            x = Robot(random.random()*Field().w_width, random.random()*Field().w_length )
            #x.set_noise(forward_noise, turn_noise, 0)
            self.p.append(x)  
            
    def step(self):
        
        self.myrobot = self.myrobot.move(0, 0.02)
        z = self.myrobot.sense()

        # now we simulate a robot motion for each of
        # these particles
        p_tmp = []
        p = self.p
        for i in range(self.n):
            p_tmp.append(p[i].move(0, 0.02))
        self.p = p_tmp
        return p_tmp
    
    def do_n_steps(self, steps):
        for i in range(steps):
            self.step()
            
    def resampling(self):
        p_tmp = []
        w = []
        for i in range(self.n):
            z = self.myrobot.sense()
            w.append(self.p[i].measurement_prob(z))
        
        index = int(random.random() * self.n)
        beta = 0.0
        mw = max(w)
        for i in range(self.n):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % self.n
            p_tmp.append(self.p[index])
            
        self.p = p_tmp
        return w ,p_tmp
    
    
def test_filter(N, steps, folder_name):
    landmarks = []
    f = Field()
    points = f.right_line_points()
    landmarks.extend(list(zip(points[0], points[1])))
    points = f.left_line_points()
    landmarks.extend(list(zip(points[0], points[1])))
    points = f.low_line_points()
    landmarks.extend(list(zip(points[0], points[1])))
    points = f.hight_line_points()
    landmarks.extend(list(zip(points[0], points[1])))
    myrobot = Robot()
    
    pf = ParticleFilter(myrobot, N)
    field = Field()
    for i in range(100):
        p = pf.step()
        w, pr = pf.resampling()
        field.show_field(pf.myrobot, i, p, pr, w, folder_name)
