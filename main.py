from rover_domain import Task_Rovers


class Parameters:
    def __init__(self):


        #Rover domain
        self.dim_x = self.dim_y = 15 #HOW BIG IS THE ROVER WORLD
        self.obs_radius = 15 #OBSERVABILITY FOR EACH ROVER
        self.act_dist = 1.5 #DISTANCE AT WHICH A POI IS CONSIDERED ACTIVATED (OBSERVED) BY A ROVER
        self.angle_res = 30 #ANGLE RESOLUTION OF THE QUASI-SENSOR THAT FEEDS THE OBSERVATION VECTOR
        self.num_poi = 10 #NUM OF POIS
        self.num_rover = 4 #NUM OF ROVERS
        self.num_timestep = 25 #TIMESTEP PER EPISODE
        self.poi_rand = False #IS THE POI INITIALIZED RANDOMLY?
        self.coupling = 2 #AMOUNT OF JOINT CONCURRENT ROVERS REQURED TO OBSERVE A POI
        self.rover_speed = 1 #MAX SPEED OF THE ROVER
        self.render = 'False' #RENDER THE ENV?
        self.sensor_model = 2 #1: STATE MEASURES DENSITY
                              #2: STATE MEASURES DISTANCE TO CLOSEST ROVER/POI


        #Dependents
        self.state_dim = 2*360 / self.angle_res + 5
        self.action_dim = 2

        #LEARNING
        self.num_episodes = 100000


#Parameter class
args = Parameters()

#Define the gym-like Env
env = Task_Rovers(args)

#MAIN LEARNING LOOP
for i_episode in range(args.num_episodes):

    joint_state = env.reset()

    #ONE EPISODE OF SIMULATION
    for t in range(args.num_timestep):

        #PUT YOUR POLICY HERE: JOINT_STATE --> JOINT ACTION
        joint_action = env.sample_action()

        #FORWARD THE ENV ONE STEP INTO THE FUTURE
        joint_next_state, joint_reward, done, info = env.step(joint_action)

        #CONTINUE WITH THE EPISODE
        joint_state = joint_next_state

        #Low-key Visualize
        env.visualize()





