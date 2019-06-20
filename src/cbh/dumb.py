import os
import re

import pandas as pd
import requests

def get_xkcd_colors():
    r = requests.get('https://xkcd.com/color/rgb.txt')
    text = r.text
    new = re.split(r'\t+', text.rstrip('\t'))

    with open("tmp.txt", "w") as text_file:
        text_file.write(text)

    df = pd.read_csv("tmp.txt", sep='\t',names=['color' , "hex", "misc"] )
    # print(df)
    xkcd_color_names = df["color"][1:].tolist()
    xkcd_color_hex = df["hex"][1:].tolist()
    # print(xkcd_color_names)
    # print(xkcd_color_hex)\
    os.remove("tmp.txt")
    return {'hex':xkcd_color_hex, 'names':xkcd_color_names}
    
colorz = get_xkcd_colors()

# print(colorz['hex'])
# print(colorz['names'])
