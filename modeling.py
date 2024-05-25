import os
from PIL import Image

in_directory = r'E:\E2CR\output\R. 317 (3)\text_words'
out_directory = r'E:\E2CR\modeling\training\words'

print(f"directory = {in_directory}")

csv_file = open(os.path.join(out_directory, "ground_truth.csv"), 'w') 

for filename in os.listdir(in_directory):
    if filename.endswith(".jpg"):
        print(f"processing file: {filename}")
        image = Image.open(os.path.join(in_directory, filename))
        image = image.convert('1') 
        image.save(os.path.join(out_directory, filename))
        csv_file.write(filename + '\n')
csv_file.close
        
        