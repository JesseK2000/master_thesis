import os
import xml.etree.ElementTree as ET

# Directory containing the XML files
xml_dir = "/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART2.2/Annotations"

# List to store the image names
image_numbers = []

# Iterate over the XML files
for filename in os.listdir(xml_dir):
    if filename.endswith(".xml"):
        xml_path = os.path.join(xml_dir, filename)
        
        try:
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract the image name
            image_name = root.find("filename").text
            
            # Extract the image number
            image_number = image_name.split(".")[0]
            
            # Add the image number to the list
            image_numbers.append(image_number)
            
        except ET.ParseError:
            # Handle the parsing error or skip the file
            
            continue


trainval_path = "/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART2.2/ImageSets/Main/test.txt"

# Read the content of trainval.txt
with open(trainval_path, "r") as file:
    trainval_content = file.readlines()

# Remove the newlines and get the image numbers from trainval.txt
trainval_image_numbers = [line.strip() for line in trainval_content]

# Filter out the image numbers that are not in the image_numbers list
filtered_trainval_image_numbers = [num for num in trainval_image_numbers if num in image_numbers]

# Write the filtered image numbers back to trainval.txt
with open(trainval_path, "w") as file:
    file.write("\n".join(filtered_trainval_image_numbers))

