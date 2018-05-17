import pandas as pd
import numpy as np
from colour import Color

def import_data(csv_path="color-data.csv"):
    color_data = pd.read_csv(csv_path)
    color_data = color_data.reindex(np.random.permutation(color_data.index))
    return color_data

def normalize_data(color_data):
    red = color_data['hex'].apply(lambda hex: get_red(hex))
    color_data['red'] = red

    green = color_data['hex'].apply(lambda hex: get_green(hex))
    color_data['green'] = green

    blue = color_data['hex'].apply(lambda hex: get_blue(hex))
    color_data['blue'] = blue

    return color_data

def get_red(hex):
    return Color(hex).red

def get_green(hex):
    return Color(hex).green

def get_blue(hex):
    return Color(hex).blue

def run():
    print("Importing color data...")
    color_data = import_data()
    print("Normalizing color data...")
    color_data = normalize_data(color_data)

    print(color_data.head(10))

run()
