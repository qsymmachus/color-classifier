import pandas as pd
import numpy as np
from colour import Color

def import_data(csv_path="color-data.csv"):
    """Imports color data from a two-field CSV (see 'color-data.csv') and returns a Dataframe."""
    color_data = pd.read_csv(csv_path)
    color_data = color_data.reindex(np.random.permutation(color_data.index))
    return color_data

def normalize_data(color_data):
    """Converts the RGB hex series in our color dataframe into a red, green, and blue series."""
    red = color_data['hex'].apply(lambda hex: get_red(hex))
    color_data['red'] = red

    green = color_data['hex'].apply(lambda hex: get_green(hex))
    color_data['green'] = green

    blue = color_data['hex'].apply(lambda hex: get_blue(hex))
    color_data['blue'] = blue

    return color_data

def get_red(hex):
    """Given a RGB hex string, returns the 'red' value as a float between 0 and 1.0"""
    return Color(hex).red

def get_green(hex):
    """Given a RGB hex string, returns the 'green' value as a float between 0 and 1.0"""
    return Color(hex).green

def get_blue(hex):
    """Given a RGB hex string, returns the 'blue' value as a float between 0 and 1.0"""
    return Color(hex).blue

def run():
    print("Importing color data...")
    color_data = import_data()
    print("Normalizing color data...")
    color_data = normalize_data(color_data)

    print(color_data.head(10))

run()
