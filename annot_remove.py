# For part 2

# import os
# import xml.etree.ElementTree as ET

# # Define the folder path where the modified annotations files are located
# folder_path = "/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART2.2/Annotations"

# # List of classes to check for
# target_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'chair', 'cow', 
#     'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 
#     'sofa', 'train', 'tvmonitor']

# # Iterate over each file in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".xml"):
#         file_path = os.path.join(folder_path, filename)
        
#         # Parse the XML file
#         tree = ET.parse(file_path)
#         root = tree.getroot()
        
#         # List to store XML elements to be removed
#         elements_to_remove = []
        
#         # Iterate over each object in the XML file
#         for obj in root.findall("object"):
#             # Extract the class label
#             class_label = obj.find("name").text
            
#             # Check if the class label is not in the target_classes list
#             if class_label not in target_classes:
#                 elements_to_remove.append(obj)
        
#         # Remove the unwanted XML elements
#         for element in elements_to_remove:
#             root.remove(element)
        
#         # Save the modified XML file
#         tree.write(file_path)


# Part 1

import os
import xml.etree.ElementTree as ET

# Define the folder path where the modified annotations files are located
folder_path = "/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART2.2/Annotations"

# List of classes to check for
target_classes = ['car', 'cat', 'horse']

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xml"):
        file_path = os.path.join(folder_path, filename)
        
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # List to store XML elements to be removed
        elements_to_remove = []
        
        # Iterate over each object in the XML file
        for obj in root.findall("object"):
            # Extract the class label
            class_label = obj.find("name").text
            
            # Check if the class label is not in the target_classes list
            if class_label not in target_classes:
                elements_to_remove.append(obj)
        
        # Remove the unwanted XML elements
        for element in elements_to_remove:
            root.remove(element)
        
        # Save the modified XML file
        tree.write(file_path)