'''Example:
Script Name: main.py /.utils.py

Author: David J. Cartwright]
Date: 5/25/2024
Last Modification Date:
Description:

Dependencies:
    - os (standard library)
'''
from common import *
from segmentation import *

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Process a given path and extract information.")
    usage = "python %(prog)s <path> [-x | --Extract] [-a | --Additional]"
    parser = argparse.ArgumentParser(description="Process a given path and perform various actions.",
                                     usage=usage)
    parser.add_argument('path', nargs='?', default=SEG_DEFAULT.INPUT_PATH, type=str, help='The path to be processed (default: ./default/path) Can be a folder or file')
    parser.add_argument('-x', '--Extract', action='store_true', help='Flag to extract information')
    parser.add_argument('-a', '--Additional', action='store_true', help='Flag to perform additional action')
 
    args = parser.parse_args()
 


    if args.Extract:
        pass

    if args.Additional:
        pass

    pathExists = os.path.exists(args.path)
    if os.path.isfile(args.path):
        print(f"Process one file {args.path}")
    else:
        print(f"Process one folder; {args.path}")
        segmentFiles(args.path)

if __name__ == "__main__":
    main()