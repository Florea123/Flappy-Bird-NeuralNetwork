import flappy_bird_gymnasium
import gymnasium
import torch
import time
import os
from agent import DQNAgent

def main():
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, device)
    
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print("Încărcăm modelul antrenat...")
        agent.load(model_path)
    else:
        print("Modelul antrenat nu a fost găsit! Agentul va juca la întâmplare.")
        print("Rulează 'python train.py' înainte pentru a antrena o rețea neuronală.")

    for episode in range(5):
        obs, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.act(obs, evaluate=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if 'score' in info:
                score = info['score']
                

            time.sleep(1/30)
            
        print(f"Joc terminat! Scor: {score}")

    env.close()

if __name__ == "__main__":
    main()