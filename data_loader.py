import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from pdb import set_trace
import requests
import math
from pathlib import Path

from geopy.geocoders import Nominatim

def get_coords(location: str):
    geolocator = Nominatim(user_agent="Chromium")
    location = geolocator.geocode(location)
    lat, lon = location.latitude, location.longitude
    
    return lat, lon 

def main():
	pass

if __name__ == '__main__':
	main()