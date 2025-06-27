import os
import shutil


def merge_folders(source_folder, destination_folder):
    #
    for root, dirs, files in os.walk(source_folder):
        #
        for file in files:
            source_path = os.path.join(root, file)  #
            destination_path = os.path.join(destination_folder, file)  #
            shutil.copy(source_path, destination_path)  #


#
source_folder = r"C:\Users\haipeng\Downloads\archive\kaggle_3m"  #
destination_folder = r"C:\Users\haipeng\Downloads\archive\after"  #
merge_folders(source_folder, destination_folder)
