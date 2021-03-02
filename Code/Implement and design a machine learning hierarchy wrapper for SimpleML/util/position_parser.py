import os


def extract_entity(stmt, ind, start, end):
    """
    Extracts entities related to position annotation.

    Parameters:
        stmt (list of str): List of strings from which entities have to be extracted.
        ind (int): Current index of the string.
        start (str): String to be matched with, in the start.
        end (str): String to be matched with, in the end.


    Returns:
        str: Extracted entity value.

    """
    return stmt[ind][stmt[ind].rfind(start) + len(start): stmt[ind].rfind(end)]


def read_pos_from_ontology(ontology_file):
    """
    Reads position annotation directly from the ontology file.

    Note:
        Output is of the format: {classname: {function_name: {function_param: position}}}

    Parameters:
        ontology_file (str): Ontology file path.

    Returns:
        dict: Nested dict of position values.

    """
    file = open(ontology_file, "r")
    content = file.readlines()
    pos_dict = {}
    for i, line in enumerate(content):
        if "<owl:Axiom>" in line:
            class_name = extract_entity(content, i + 1, "#", "\"")
            func_name = extract_entity(content, i + 5, "#", "\"")
            param_name = extract_entity(content, i + 6, "#", "\"").split("__")[-1]
            pos = float(extract_entity(content, i + 9, "\">", "</pos>"))
            if pos_dict.get(class_name):
                if pos_dict[class_name].get(func_name):
                    pos_dict[class_name][func_name].update({param_name: pos})
                else:
                    pos_dict[class_name][func_name] = {param_name: pos}
            else:
                entry = {class_name: {func_name: {param_name: pos}}}
                pos_dict.update(entry)
    print(pos_dict)
    return pos_dict


def test():
    read_pos_from_ontology(os.path.join(os.pardir, "ml_algorithms.owl"))


if __name__ == "__main__":
    test()
