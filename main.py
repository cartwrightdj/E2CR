import argparse
import cv2
import os
from common import logging, logger, Parameters, DEBUG, DEBUG_FOLDER, Statistics,  stats, Colors
from imageman import pre_process_image, postProcessSegment,extract_contours
from imageutils import get_image_stats
from segmentation import segment_image
import glob



parser = argparse.ArgumentParser(
                    prog='Early English Cursive Recognition',
                    description='Segment Scanned 18th Cenruty Documnets, OCR, adn transcribe',
                    epilog='Text at the bottom of help')

parser.add_argument("path",help="Path to image of folder of images to be processed")
parser.add_argument("--So",help="Segment Only")
parser.add_argument('--verbose', '-v',help="Verbose level" ,action='count', default=0,required=False)
parser.add_argument('--params','-p',  help="Use Parameter Preset" ,nargs='?',required=False)
parser.add_argument('--Params','-P',  help="Show Default Parameters Preset" , action='store_true')
parser.add_argument('--Debug','-D',  help="Turn on Debugging, Implies TRACE Verbose Level" , action='store_true')


action = parser.add_mutually_exclusive_group()
action.add_argument('--PreProcess','-Pp', action='store_true')
action.add_argument('--Segment','-S', action='store_true')
action.add_argument('--Train','-T', action='store_true')

args = parser.parse_args()


if args.Params: print(Parameters.print_params())
if args.verbose: logging(args.verbose)
if args.Debug: 
    DEBUG = True
    logging(3)

image_file_paths = [] 
if os.path.isdir(args.path): 
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    for extension in image_extensions:
        image_file_paths.extend(glob.glob(os.path.join(args.path, extension)))
    logger.info(f"Found {len(image_file_paths)} image files in {args.path}")  
if os.path.isfile(args.path):
    image_file_paths.append(args.path)

if args.Segment:
    logger.info(f"Segmentation Only on: '{args.path}'")

for image_file_path in image_file_paths:
    image_from_file = cv2.imread(image_file_path)
    image_file_name = os.path.basename(image_file_path)
    Statistics.fileName = image_file_name
    get_image_stats(image_from_file)
    print(f"Preprocessing {image_file_name}")
    
    pre_processed_image = pre_process_image(image_from_file,os.path.basename(image_file_path))
 
    segmented_image, row_segmentations = segment_image(pre_processed_image)
    print(f"rows: {len(segmented_image)}")
    for r, segmented_row in enumerate(row_segmentations):
        if not segmented_row is None:
            cv2.imwrite(os.path.join(os.getcwd(), './segmentation', f'row_{image_file_name}_{r:03d}.tiff'), segmented_row )
        

    for r, row in enumerate(segmented_image):
        print(f"words: {len(row)}")
        for w, word in enumerate(row):
            word = postProcessSegment(word)
            if not word is None:
                extracted_contours = extract_contours(word)
                for c, contour in enumerate(extracted_contours):
                    #cv2.imwrite(os.path.join(os.getcwd(), './segmentation', f'seg_{image_file_name}_{r:03d}_{w:03d}_{c:03d}.tiff'), contour )
                    pass
                    
            else:
                print(f"{image_file_name} line {r} word {w} was empty")
            if not word is None:
                cv2.imwrite(os.path.join(os.getcwd(), './segmentation', f'seg_{image_file_name}_{r:03d}_{w:03d}.tiff'), word )
    
    stats.append(Statistics)

for stat in stats:
    stat.display()