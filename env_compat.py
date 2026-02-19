import random

import numpy as np
import torch


def import_gym():
    try:
        import gymnasium as gym  # type: ignore
    except ImportError:
        import gym  # type: ignore
    return gym


GYM = import_gym()


def make_carracing_env(render_mode=None):
    env_ids = ("CarRacing-v3", "CarRacing-v2", "CarRacing-v1", "CarRacing-v0")
    last_error = None
    for env_id in env_ids:
        try:
            if render_mode is None:
                return GYM.make(env_id)
            return GYM.make(env_id, render_mode=render_mode)
        except TypeError:
            try:
                return GYM.make(env_id)
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to create CarRacing environment")


def reset_env(env, seed=None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()
            if hasattr(env, "seed"):
                env.seed(seed)

    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    return result


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
