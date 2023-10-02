import os

def add_suffix_to_text_files(folder_path, suffix):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and rename text files
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            new_file_name = file_name[:-4] + suffix + '.txt'  # Remove '.txt' extension and add the suffix
            new_file_path = os.path.join(folder_path, new_file_name)

            os.rename(file_path, new_file_path)
            print(f"Renamed '{file_name}' to '{new_file_name}'.")

# Specify the folder path and the suffix to add
folder_path = '/data/Jiale/ASTERS/datasets/shooter/train/labels'
suffix = '_aug'

# Call the function to add the suffix to text files in the folder
add_suffix_to_text_files(folder_path, suffix)
