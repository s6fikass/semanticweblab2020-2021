import os
import numpy as np

from pytorch.load.kg import KG
from pytorch.load.read import generate_sharing_id, generate_mapping_id, generate_sup_relation_triples, generate_sup_attribute_triples
from pytorch.load.read import uris_relation_triple_2ids, uris_attribute_triple_2ids, uris_pair_2ids, read_relation_triples, read_attribute_triples, read_links


class KGs:

    def __init__(self, kg1: KG, kg2: KG, train_links, valid_links=None, test_links=None, mode='mapping', ordered=True):
        if mode == "sharing":
            ent_ids1, ent_ids2 = generate_sharing_id(train_links, kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_sharing_id([], kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_sharing_id([], kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        else:
            ent_ids1, ent_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)

        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2)

        train_links = uris_pair_2ids(train_links, ent_ids1, ent_ids2)
        valid_links = uris_pair_2ids(valid_links, ent_ids1, ent_ids2) if valid_links is not None else []
        test_links = uris_pair_2ids(test_links, ent_ids1, ent_ids2) if test_links is not None else []

        if mode == 'swapping':
            sup_triples1, sup_triples2 = generate_sup_relation_triples(train_links,
                                                                       kg1.rt_dict, kg1.hr_dict,
                                                                       kg2.rt_dict, kg2.hr_dict)
            kg1.add_sup_relation_triples(sup_triples1)
            kg2.add_sup_relation_triples(sup_triples2)

            sup_triples1, sup_triples2 = generate_sup_attribute_triples(train_links, kg1.av_dict, kg2.av_dict)
            kg1.add_sup_attribute_triples(sup_triples1)
            kg2.add_sup_attribute_triples(sup_triples2)

        self.kg1 = kg1
        self.kg2 = kg2

        self.num_train_entities = len(train_links)
        self.num_valid_entities = len(valid_links)
        self.num_test_entities = len(test_links)
        self.all_entities = np.array(train_links + valid_links + test_links, dtype=np.int32)

        self.num_entities = len(self.kg1.entities_set | self.kg2.entities_set)
        self.num_relations = len(self.kg1.relations_set | self.kg2.relations_set)
        self.num_attributes = len(self.kg1.attributes_set | self.kg2.attributes_set)

    def get_entities(self, split, kg=None):
        """
        Get entities from the selected kg and split.

        Parameters
        ----------
        split
            Name of the data split.
        kg
            The KG identity either 1 or 2.
        """
        if split == 'train':
            entities = self.all_entities[:self.num_train_entities]
        elif split == 'valid':
            entities = self.all_entities[self.num_train_entities:self.num_train_entities + self.num_valid_entities]
        elif split == 'test':
            entities = self.all_entities[-self.num_test_entities:]
        else:  # valid and test
            entities = self.all_entities[self.num_train_entities:]
        return entities if kg is None else entities[:, kg - 1]


def read_kgs(path, division, mode, ordered, remove_unlinked=False):
    """
    Read kgs from specified path.

    Parameters
    ----------
    path
        Path that contains the data files.
    division
        Path to the dataset division directory.
    mode
        Mode to initialize the `KGs` class instance with.
    ordered
        Ordered mode for the `KGs` class instance.
    remove_unlinked
        Flag to remove unlinked triples.
    """
    kg1_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'rel_triples_1'))
    kg2_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'rel_triples_2'))
    kg1_attribute_triples, _, _ = read_attribute_triples(os.path.join(path, 'attr_triples_1'))
    kg2_attribute_triples, _, _ = read_attribute_triples(os.path.join(path, 'attr_triples_2'))

    train_links = read_links(os.path.join(path, division, 'train_links'))
    valid_links = read_links(os.path.join(path, division, 'valid_links'))
    test_links = read_links(os.path.join(path, division, 'test_links'))

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, valid_links, test_links, mode, ordered)
    return kgs


def read_reversed_kgs(path, division, mode, ordered, remove_unlinked=False):
    kg1_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'rel_triples_1'))
    kg2_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'rel_triples_2'))
    kg1_attribute_triples, _, _ = read_attribute_triples(os.path.join(path, 'attr_triples_1'))
    kg2_attribute_triples, _, _ = read_attribute_triples(os.path.join(path, 'attr_triples_2'))

    train_links = [(j, i) for i, j in read_links(os.path.join(path, division, 'train_links'))]
    valid_links = [(j, i) for i, j in read_links(os.path.join(path, division, 'valid_links'))]
    test_links = [(j, i) for i, j in read_links(os.path.join(path, division, 'test_links'))]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, valid_links, test_links, mode, ordered)
    return kgs


def read_kgs_dbp_or_dwy(path, division, mode, ordered, remove_unlinked=False):
    path = os.path.join(path, division)
    kg1_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'triples_1'))
    kg2_relation_triples, _, _ = read_relation_triples(os.path.join(path, 'triples_2'))
    if os.path.exists(os.path.join(path, 'sup_pairs')):
        train_links = read_links(os.path.join(path, 'sup_pairs'))
    else:
        train_links = read_links(os.path.join(path, 'sup_ent_ids'))
    if os.path.exists(os.path.join(path, 'ref_pairs')):
        test_links = read_links(os.path.join(path, 'ref_pairs'))
    else:
        test_links = read_links(os.path.join(path, 'ref_ent_ids'))
    print()
    if remove_unlinked:
        for i in range(10000):
            print("removing times:", i)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n1 = len(kg1_relation_triples)
            n2 = len(kg2_relation_triples)
            train_links, test_links = remove_no_triples_link(kg1_relation_triples, kg2_relation_triples, train_links, test_links)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n11 = len(kg1_relation_triples)
            n22 = len(kg2_relation_triples)
            if n1 == n11 and n2 == n22:
                break
            print()

    kg1 = KG(kg1_relation_triples, [])
    kg2 = KG(kg2_relation_triples, [])
    kgs = KGs(kg1, kg2, train_links, test_links=test_links, mode=mode, ordered=ordered)
    return kgs


def remove_no_triples_link(kg1_relation_triples, kg2_relation_triples, train_links, test_links):
    kg1_entities, kg2_entities = set(), set()
    for h, r, t in kg1_relation_triples:
        kg1_entities.add(h)
        kg1_entities.add(t)
    for h, r, t in kg2_relation_triples:
        kg2_entities.add(h)
        kg2_entities.add(t)
    print("before removing links with no triples:", len(train_links), len(test_links))
    new_train_links, new_test_links = set(), set()
    for i, j in train_links:
        if i in kg1_entities and j in kg2_entities:
            new_train_links.add((i, j))
    for i, j in test_links:
        if i in kg1_entities and j in kg2_entities:
            new_test_links.add((i, j))
    print("after removing links with no triples:", len(new_train_links), len(new_test_links))
    return list(new_train_links), list(new_test_links)


def remove_unlinked_triples(triples, links):
    print("before removing unlinked triples:", len(triples))
    linked_entities = set()
    for i, j in links:
        linked_entities.add(i)
        linked_entities.add(j)
    linked_triples = set()
    for h, r, t in triples:
        if h in linked_entities and t in linked_entities:
            linked_triples.add((h, r, t))
    print("after removing unlinked triples:", len(linked_triples))
    return linked_triples
