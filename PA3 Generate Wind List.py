# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:14:59 2025

@author: marti
"""
# -*- coding: utf-8 -*-
"""
wind_generator.py

Generates a random list of wind magnitudes and angles, then saves them.
"""

import math
import random
import pickle  # or json, or numpy, etc.

def generate_wind_list(
        init_dir_deg=30.0,
        dir_range_deg=90.0,
        wind_mag_min=0.0,
        wind_mag_max=0.5,
        gauss_sigma=0.05,
        num_wind_steps=3000,
        seed=9999):
    """
    Returns a list of (magnitude, angle) pairs corresponding to a noisy,
    clamped wind signal.  Angles in radians, magnitudes in (e.g.) m/s.

    - init_dir_deg:   initial direction in degrees
    - dir_range_deg:  total +/- range around init_dir, e.g. 90 => ±45 deg
    - wind_mag_min:   clamp minimum wind magnitude
    - wind_mag_max:   clamp maximum wind magnitude
    - gauss_sigma:    standard deviation for random steps
    - num_wind_steps: number of frames to generate
    - seed:           random seed
    """
    rng = random.Random(seed)

    # Convert initial direction/range to radians
    dir_rad = math.radians(init_dir_deg)
    half_range = math.radians(dir_range_deg / 2.0)

    # Start at mid-range magnitude
    mag = 0.5 * (wind_mag_min + wind_mag_max)
    angle = dir_rad

    wind_data = []
    for _ in range(num_wind_steps):
        # random step in magnitude
        dm = rng.gauss(0.0, gauss_sigma)
        mag += dm
        mag = max(wind_mag_min, min(wind_mag_max, mag))

        # random step in angle
        dtheta = rng.gauss(0.0, gauss_sigma)
        angle += dtheta
        # clamp angle to [dir_rad - half_range, dir_rad + half_range]
        lo = dir_rad - half_range
        hi = dir_rad + half_range
        if angle < lo:
            angle = lo
        if angle > hi:
            angle = hi

        wind_data.append((mag, angle))

    return wind_data


if __name__ == "__main__":
    # Generate the list
    wdata = generate_wind_list(
        init_dir_deg=30.0,
        dir_range_deg=90.0,
        wind_mag_min=0.0,
        wind_mag_max=0.5,
        gauss_sigma=0.05,
        num_wind_steps=3000,
        seed=9999
    )
    print(f"Generated wind_data with length = {len(wdata)}")
    
    # Option 1: Print the first few elements
    # print(wdata[:10])

    # Option 2: Save to a file using pickle
    with open("wind_data.pkl", "wb") as f:
        pickle.dump(wdata, f)
    print("wind_data saved to wind_data.pkl")
