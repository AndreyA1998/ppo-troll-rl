import numpy as np
import platform

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers.rendering import RecordVideo

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
    
def apply_gray_scale_wrap(env):
    # With the argument values chosen as below, the gym.wrappers.AtariPreprocessing wrapper
    # only converts images to grayscale and downsamples them the screen_size
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=0, # the default value 30 can be harmful with FireResetEnv and frame_skip=5
        frame_skip=1, # frame_skip has already been set to 5 inside the env
        terminal_on_life_loss=False, # we do this explicitly in the FireResetEnv wrapper
        screen_size=84, # please use 84 (which is the standard value) or 64 (which will save some computations and memory)
    )
    return env
    
def apply_atary_specific_wrap(env, monitor=False, episodic_life=True):
    if monitor:
        env = RecordVideo(env, video_folder="./bo_videos", fps=30)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    return env
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.scale = 255.0
        orig_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=self.observation(orig_obs_space.low),
            high=self.observation(orig_obs_space.high),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / self.scale
    
def get_mono_bo_env(monitor: bool = False, episodic_life: bool = True, apply_frame_stack: bool = True) -> gym.Env:
    """
    Builds the environment with all the wrappers applied.
    The environment is meant be used directly as an RL algorithm input.

    apply_frame_stack=False can be useful for vecotrized environments, which are not required for this assignment.
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = apply_gray_scale_wrap(env)
    env = ScaledFloatFrame(env)
    env = apply_atary_specific_wrap(env, monitor=monitor, episodic_life=episodic_life)
    if apply_frame_stack:
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

def get_bo_env(batch_size: int = 1) -> gym.Env:
    env_fns = [lambda: get_mono_bo_env(apply_frame_stack=False) for _ in range(batch_size)]
    if platform.system() == "Windows":
        vec_env = SyncVectorEnv(env_fns)
    else:
        vec_env = AsyncVectorEnv(env_fns)
    
    # custom wrapper for frame stacking
    class FrameStack:
        def __init__(self, env, stack_size):
            self.env = env
            self.stack_size = stack_size
            self.num_envs = env.num_envs
            self.single_observation_space = env.single_observation_space
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(stack_size, *env.single_observation_space.shape),
                dtype=env.single_observation_space.dtype,
            )
            self.action_space = env.action_space
            self.frames = np.zeros(
                (self.num_envs, stack_size, *env.single_observation_space.shape),
                dtype=env.single_observation_space.dtype
            )
            
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.frames[:] = np.repeat(obs[:, None, ...], self.stack_size, axis=1)
            return self.frames, info
            
        def step(self, actions):
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            self.frames = np.roll(self.frames, shift=-1, axis=1)
            self.frames[:, -1] = obs
            return self.frames, rewards, terminated, truncated, info
            
        def __getattr__(self, name):
            return getattr(self.env, name)
        
    vec_env = FrameStack(vec_env, 4)
    
    return vec_env