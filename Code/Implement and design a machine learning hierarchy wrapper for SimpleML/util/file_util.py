import os


def create_and_write_file(filename, content):
    """
    Creates and writes contents into a file.

    Parameters:
        filename (str): Name of the file.
        content (str): Content which has to be written into a file.

    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    f.write(content)
    f.close()


def create_folders_and_subfolders(folder_path):
    """
    Creates folders and subfolders in a specific path.

    Parameters:
        folder_path (str): Path of the folder.

    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
