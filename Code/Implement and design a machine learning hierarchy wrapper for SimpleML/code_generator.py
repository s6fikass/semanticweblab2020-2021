from owlready2 import *
import os
import shutil

from util.file_content_creator import create_file_contents
from util import file_util
from util.position_parser import read_pos_from_ontology


def get_file_path(node, file, root_class):
    """
    Generates file path based on class.

    Parameters:
        node (object of owlready2.entity.ThingClass): Current class for which file path has to be created.
        file (str): Base file path.
        root_class (object of owlready2.entity.ThingClass): Root class of the tree.

    Returns:
        str: File path of the current class.

    """
    file_path = file
    if node == root_class:
        file_path = os.path.join(file, file)
    index = file_path.rfind(os.path.sep)
    return file_path[:index+1] + "_" + file_path[index+1:] + ".py"


def main():
    child_parent_map = {}
    ontology_file = "ml_algorithms.owl"
    pos_dict = read_pos_from_ontology(ontology_file)
    onto = get_ontology(ontology_file).load()
    queue = []
    root_class = None
    print([i for i in onto.annotation_properties()])
    print(list(onto.classes()))
    for onto_class in list(onto.classes()):
        if onto_class.label[0] == 'MLalgorithms':
            root_class = onto_class

    #shutil.rmtree("MLalgorithms")
    queue.append(root_class)
    dir_structure = [root_class.label[0]]
    child_parent_map[root_class] = None
    while queue:
        node = queue.pop(0)
        file = dir_structure.pop(0)
        if onto.get_children_of(node):
            file_util.create_folders_and_subfolders(file)
        content = create_file_contents(file, node, child_parent_map, pos_dict)
        file_util.create_and_write_file(
            get_file_path(node, file, root_class), content)
        for child in onto.get_children_of(node):
            queue.append(child)
            child_parent_map[child] = node
            dir_structure.append(os.path.join(file, child.label[0]))


if __name__ == "__main__":
    main()
