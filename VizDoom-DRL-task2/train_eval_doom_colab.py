import os
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
from datetime import datetime
import json
from doom_agents import (
    CNNPolicy, 
    TransformerPolicy, 
    HybridCNNTransformerPolicy, 
    create_doom_env
)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

class ExperimentManager:
    def __init__(
        self,
        config_path="basic.cfg",
        n_envs=8,
        n_seeds=3,
        total_timesteps=1_000_000,
        eval_episodes=20,
        frame_skip=4
    ):
        self.config_path = config_path
        self.n_envs = n_envs
        self.n_seeds = n_seeds
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self.frame_skip = frame_skip
        
        # Training hyperparameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.clip_range = 0.2
        
        # Setup logging directories in Google Drive
        self.base_dir = '/content/gdrive/MyDrive/ML4_vizdoom'
        for dir_name in ['logs', 'models', 'videos']:
            path = os.path.join(self.base_dir, dir_name)
            os.makedirs(path, exist_ok=True)
        
        # Available policy architectures
        self.policies = {
            "cnn": CNNPolicy,
            "transformer": TransformerPolicy,
            "hybrid": HybridCNNTransformerPolicy
        }
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def make_env(self, seed=None, capture_video=False):
        """Create vectorized environment"""
        def make_env_fn():
            def _init():
                env = create_doom_env(self.config_path, self.frame_skip)
                if seed is not None:
                    env.seed(seed)
                    env.action_space.seed(seed)
                env = Monitor(env, os.path.join(self.base_dir, "logs", f"seed_{seed}" if seed is not None else "tmp"))
                return env
            return _init
            
        env = DummyVecEnv([make_env_fn() for _ in range(self.n_envs)])
        
        if capture_video:
            env = VecVideoRecorder(
                env,
                os.path.join(self.base_dir, "videos"),
                record_video_trigger=lambda step: step % 10000 == 0,
                video_length=200
            )
            
        return env
        
    def create_agent(self, policy_type, env):
        """Create PPO agent with specified policy"""
        policy_class = self.policies[policy_type]
        
        policy_kwargs = dict(
            features_extractor_class=policy_class,
            features_extractor_kwargs=dict(features_dim=512),
            device=self.device
        )
        
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=1
        )
        
        return model
        
    def evaluate_agent(self, model, env):
        """Evaluate agent over multiple episodes"""
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        
        obs = env.reset()
        
        for _ in range(self.eval_episodes):
            done = False
            total_reward = 0
            steps = 0
            start_time = datetime.now()
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]  # Only track first environment
                steps += 1
                if done[0]:  # Check first environment
                    obs = env.reset()
                    break
                    
            episode_time = (datetime.now() - start_time).total_seconds()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_times.append(episode_time)
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'mean_episode_time': float(np.mean(episode_times)),
            'std_episode_time': float(np.std(episode_times))
        }
        
    def train_and_evaluate(self):
        """Train and evaluate all policy types with multiple seeds"""
        results = {}
        
        for policy_type in self.policies:
            policy_results = []
            
            for seed in range(self.n_seeds):
                print(f"\nTraining {policy_type} policy (Seed {seed + 1}/{self.n_seeds})")
                
                # Initialize wandb
                run = wandb.init(
                    project="vizdoom-rl",
                    name=f"{policy_type}-seed{seed}",
                    config={
                        "policy": policy_type,
                        "seed": seed,
                        "total_timesteps": self.total_timesteps,
                        "n_envs": self.n_envs,
                        "learning_rate": self.learning_rate,
                        "n_steps": self.n_steps,
                        "batch_size": self.batch_size,
                        "n_epochs": self.n_epochs,
                        "clip_range": self.clip_range,
                        "device": str(self.device)
                    },
                    reinit=True
                )
                
                # Set random seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # Create environment and model
                env = self.make_env(seed=seed, capture_video=True)
                model = self.create_agent(policy_type, env)
                
                # Train model
                start_time = time.time()
                try:
                    model.learn(
                        total_timesteps=self.total_timesteps,
                        progress_bar=True
                    )
                except Exception as e:
                    print(f"Training failed: {str(e)}")
                    continue
                    
                train_time = time.time() - start_time
                
                # Save model to Google Drive
                try:
                    save_path = os.path.join(self.base_dir, "models", f"{policy_type}_seed{seed}")
                    model.save(save_path)
                except Exception as e:
                    print(f"Failed to save model: {str(e)}")
                
                # Create evaluation environment
                eval_env = self.make_env(seed=seed+100)  # Different seed for evaluation
                
                # Evaluate model
                eval_results = self.evaluate_agent(model, eval_env)
                eval_results['train_time'] = train_time
                policy_results.append(eval_results)
                
                # Log evaluation metrics
                wandb.log({
                    "eval_mean_reward": eval_results['mean_reward'],
                    "eval_mean_episode_length": eval_results['mean_episode_length'],
                    "eval_mean_episode_time": eval_results['mean_episode_time'],
                    "train_time": train_time
                })
                
                # Close environments
                env.close()
                eval_env.close()
                wandb.finish()
                
            # Calculate aggregate results for this policy
            if policy_results:
                aggregate_results = {
                    'mean_reward': float(np.mean([r['mean_reward'] for r in policy_results])),
                    'std_reward': float(np.std([r['mean_reward'] for r in policy_results])),
                    'mean_episode_length': float(np.mean([r['mean_episode_length'] for r in policy_results])),
                    'std_episode_length': float(np.std([r['mean_episode_length'] for r in policy_results])),
                    'mean_train_time': float(np.mean([r['train_time'] for r in policy_results])),
                    'std_train_time': float(np.std([r['train_time'] for r in policy_results])),
                    'individual_seeds': policy_results
                }
                
                results[policy_type] = aggregate_results
                
                # Save policy results
                results_path = os.path.join(self.base_dir, "logs", f"{policy_type}_results.json")
                with open(results_path, 'w') as f:
                    json.dump(aggregate_results, f, indent=4)
        
        # Save overall results
        if results:
            final_results_path = os.path.join(self.base_dir, "logs", "all_results.json")
            with open(final_results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
        return results

def main():
    experiment = ExperimentManager()
    results = experiment.train_and_evaluate()
    
    # Print final results
    print("\nFinal Results:")
    for policy_type, metrics in results.items():
        print(f"\n{policy_type.upper()} Policy:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_episode_length']:.2f} ± {metrics['std_episode_length']:.2f}")
        print(f"Mean Training Time: {metrics['mean_train_time']:.2f} ± {metrics['std_train_time']:.2f}")

if __name__ == "__main__":
    main()