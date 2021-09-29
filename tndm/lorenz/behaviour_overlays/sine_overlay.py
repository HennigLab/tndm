import numpy as np


def sine_overlay(step, start, stop, scale=1, frequency=8):
    t = np.arange(start, stop, step)
    dynamics = np.sin(2 * np.pi * frequency * t) * scale
    return dynamics
