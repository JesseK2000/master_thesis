# OPEN SET PART 1

# import os
# import shutil
# import xml.etree.ElementTree as ET

# # Define the classes to include
# classes_to_include = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'chair', 'cow', 
#                     'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# # Path to the Pascal VOC dataset
# voc_dataset_dir = '/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007'
# annotation_dir = os.path.join(voc_dataset_dir, 'Annotations')
# image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')

# # Path to store the new dataset
# new_dataset_dir = '/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART1.1'
# output_annotation_dir = os.path.join(new_dataset_dir, 'Annotations')
# output_image_dir = os.path.join(new_dataset_dir, 'JPEGImages')

# # Create the new dataset directory
# os.makedirs(new_dataset_dir, exist_ok=True)
# os.makedirs(output_image_dir, exist_ok=True)
# os.makedirs(output_annotation_dir, exist_ok=True)

# # Iterate through the VOC dataset
# for root, dirs, files in os.walk(voc_dataset_dir):
#     for file in files:
#         # Check if the file is an annotation XML file
#         if file.endswith('.xml'):
#             # Parse the XML file
#             xml_path = os.path.join(root, file)
#             tree = ET.parse(xml_path)
#             root_elem = tree.getroot()

#             # Get the object annotations
#             objects = root_elem.findall('object')

#             # Check if any of the objects belong to the desired classes
#             found_class = False
#             for obj in objects:
#                 class_name = obj.find('name').text
#                 if class_name in classes_to_include:
#                     found_class = True
#                     break

#             # If any object belongs to the desired classes, copy the image and annotation
#             if found_class:
#                 # Copy the image
#                 image_filename = os.path.splitext(file)[0] + '.jpg'
#                 image_src = os.path.join(voc_dataset_dir, 'JPEGImages', image_filename)
#                 image_dst = os.path.join(new_dataset_dir, 'JPEGImages', image_filename)
#                 shutil.copy(image_src, image_dst)

#                 # Copy the annotation
#                 annotation_dst = os.path.join(new_dataset_dir, 'Annotations', file)
#                 shutil.copy(xml_path, annotation_dst)


# OPEN SET PART 2

import os
import shutil
import xml.etree.ElementTree as ET

# Define the classes to include
classes_to_include = ['car', 'cat', 'horse']

# Path to the Pascal VOC dataset
voc_dataset_dir = '/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007'
annotation_dir = os.path.join(voc_dataset_dir, 'Annotations')
image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')

# Path to store the new dataset
new_dataset_dir = '/home/jesse/thesis/second_YOLOv1_PyTorch/data/VOCdevkit/VOC2007OPENPART2.2'
output_annotation_dir = os.path.join(new_dataset_dir, 'Annotations')
output_image_dir = os.path.join(new_dataset_dir, 'JPEGImages')

# Create the new dataset directory
os.makedirs(new_dataset_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Iterate through the VOC dataset
for root, dirs, files in os.walk(voc_dataset_dir):
    for file in files:
        # Check if the file is an annotation XML file
        if file.endswith('.xml'):
            # Parse the XML file
            xml_path = os.path.join(root, file)
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()

            # Get the object annotations
            objects = root_elem.findall('object')

            # Check if any of the objects belong to the desired classes
            found_class = False
            for obj in objects:
                class_name = obj.find('name').text
                if class_name in classes_to_include:
                    found_class = True
                    break

            # If any object belongs to the desired classes, copy the image and annotation
            if found_class:
                # Copy the image
                image_filename = os.path.splitext(file)[0] + '.jpg'
                image_src = os.path.join(voc_dataset_dir, 'JPEGImages', image_filename)
                image_dst = os.path.join(new_dataset_dir, 'JPEGImages', image_filename)
                shutil.copy(image_src, image_dst)

                # Copy the annotation
                annotation_dst = os.path.join(new_dataset_dir, 'Annotations', file)
                shutil.copy(xml_path, annotation_dst)
