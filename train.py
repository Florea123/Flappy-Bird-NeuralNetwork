import gymnasium as gym
import flappy_bird_gymnasium
import torch
import numpy as np
from agent import DQNAgent

def main():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True, disable_env_checker=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, device)
    
    epochs = 10000
    max_score = 0
    
    for epoch in range(epochs):
        state, info = env.reset()
        total_reward = 0
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if terminated:
                reward = -10
            elif reward == 1.0:
                reward = 10
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if 'score' in info:
                score = info['score']
                
        
        if score > max_score and score > 0:
            max_score = score
            agent.save("best_model.pth")
            print(f"Modelul a fost salvat cu scorul: {max_score}")
            
    env.close()

if __name__ == "__main__":
    main()
