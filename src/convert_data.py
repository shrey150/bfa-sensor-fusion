import numpy as np
import pandas as pd
from math import sqrt
import sys, getopt
from os import path

import lib.csv_reader as reader
import lib.mag_calibration as mag_cal
from lib.constants import *

INPUT_FILE = None
OUTPUT_FILE = None
CALIBRATE_MAG = False
SAME_SPS = False

HELP = """
convert_data.py -i <inputfile>

Optional:
    -c to apply magnetometer calibration
    -o to set output file location
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], "ci:o:", ["input=","output=","calibrate-mag","same-sps","help"])
except getopt.GetoptError:
    print(HELP)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h","--help"):
        print(HELP)
        sys.exit(0)
    if opt in ("-i","--input"):
        INPUT_FILE = arg
    elif opt in ("-o","--output"):
        OUTPUT_FILE = arg
    elif opt in ("-c","--calibrate-mag"):
        CALIBRATE_MAG = True
    elif opt == "--same-sps":
        SAME_SPS = True

if INPUT_FILE is None:
    print("Error: input file not specified")
    sys.exit(2)

if OUTPUT_FILE is None:
    file_no_ext = path.splitext(path.basename(INPUT_FILE))[0]
    OUTPUT_FILE = f"out/{file_no_ext}_converted.csv"

print("Converting test...")

data, params = reader.read(INPUT_FILE, same_sps=SAME_SPS)

if CALIBRATE_MAG:
    print("Calibrating magnetometer data...")
    data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS])

print(f"Saving data to {OUTPUT_FILE}...")

data.to_csv(OUTPUT_FILE, index=False, header=False)

print("Done.")