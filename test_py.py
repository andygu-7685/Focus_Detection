import sys
import os

# 1. Add the folder containing your .pyd
module_path = r"C:\Users\USER\Documents\Research\Focus_Algo\build\bin"
sys.path.append(module_path)

# 2. Tell Windows where to find the OpenCV/MinGW DLL dependencies
# On Python 3.8+, you MUST use add_dll_directory
os.add_dll_directory(module_path)
os.add_dll_directory(r"C:\mingw64\bin") # Path to your MinGW bin

import my_module
print("Import successful!")
# Check if the module loaded and show available functions
print("Module Name:", my_module.__name__)
print("Available functions:", dir(my_module))

# # Test your image processor function
try:
    # Use the arguments you defined in C++
    my_module.process_folder(
        input_folder="Stack", 
        output_folder="output_images", 
        block_size=6, 
        threshold_val=180
    )
    print("Success! C++ code executed.")
except Exception as e:
    print(f"Error calling C++ function: {e}")