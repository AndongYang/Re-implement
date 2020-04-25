# coding = utf-8
import os

def get_h5_list(file_dir):   
    all_file = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.h5':  
                all_file.append(os.path.join(root, file))           
                return all_file
                
