#!/usr/local/bin/python3
# Generate random feature vectors for each diagram part

import sys
import json
import random
import pickle
import gzip
import numpy as np
import re

diagram_label_file = sys.argv[1]
vgg_dir = sys.argv[2]
matching_dir = sys.argv[3]
out_file = sys.argv[4]

def label_to_matching_vector(diagram_json, label):
    matching_vec = []
    # img_id = re.sub("-([^0-9])", "_\g<1>", j["imageId"])
    img_id = j["imageId"]
    matching_file = matching_dir + "/" + j["label"] + "/" + img_id + "_" + label + ".pklz"
    # print(matching_file)
    with open(matching_file, 'rb') as g:
        matching = pickle.loads(gzip.decompress(g.read()))

        if len(matching) == 1:
            # Choi's format
            matching_vec = matching[0]
        else:
            # Ani's format
            matching_vec = matching        

        # print(matching_vec)

    return matching_vec

    '''
    # One-hot at a label-specific index.
    DIMS = 32
    vec = [0.0] * DIMS
    h = label.__hash__() % DIMS
    vec[h] = 1.0
    return np.array(vec)
    '''

def label_to_vgg_vector(diagram_json, label, scale):
    vgg_vec = []
    vgg_file = vgg_dir + "/" + j["label"] + "/" + j["imageId"] + "_" + label + "_" + str(scale) + ".png.pkl"
    with open(vgg_file, 'rb') as g:
        vgg = pickle.loads(gzip.decompress(g.read()))
        vgg_vec = vgg[0]

    return vgg_vec

def label_to_feature_vector(label, xy, width, height):
    DIMS = 2
    vec = [0.0] * DIMS

    # X/Y coordinates normalized by image size
    vec[0] = float(xy[0]) / width
    vec[1] = float(xy[1]) / height
    return np.array(vec)

    # Random with a high-scoring element in a label-specific index.
    '''
    h = label.__hash__() % (DIMS / 2)
    vec[h] = 3.0
    for i in xrange(len(vec)):
        vec[i] += random.gauss(0.0, 1.0)
    return vec
    '''

    # One-hot at a label-specific index.
    '''
    h = label.__hash__() % DIMS
    vec[h] = 1.0
    return vec
    '''

    # Random around a mean per label    
    '''
    for i in xrange(len(vec)):
        mean_random = random.Random()
        mean_random.seed(label.__hash__() * i)
        mean = mean_random.uniform(-1, 1)
        
        vec[i] = random.gauss(mean, 1.0)
    return vec
    '''

    # Completely random
    '''
    for i in xrange(len(vec)):
        vec[i] = random.gauss(0.0, 1.0)
    return vec
    '''

image_points = {}
with open(diagram_label_file, 'r') as f:
    for line in f:
        j = json.loads(line)

        image_id = j["imageId"]
        width = j["width"]
        height = j["height"]

        if not image_id in image_points:
            image_points[image_id] = {}

        # print image_id
        for p in j["points"]:
            xy = tuple(p["xy"])
            label = p["label"]
            xy_vec = label_to_feature_vector(label, xy, width, height)
            matching_vec = label_to_matching_vector(j, label)

            # Zeroed out to keep file size down.
            # vgg_vec_0 = label_to_vgg_vector(j, label, 0)
            # vgg_vec_1 = label_to_vgg_vector(j, label, 1)
            # vgg_vec_2 = label_to_vgg_vector(j, label, 2)
            vgg_vec_0 = np.array([0])
            vgg_vec_1 = np.array([0])
            vgg_vec_2 = np.array([0])

            # print "  ", xy, label
            # print "  ", vec
            
            image_points[image_id][xy] = {"xy_vec" : xy_vec, "matching_vec" : matching_vec, "vgg_0_vec" : vgg_vec_0,
                                          "vgg_1_vec" : vgg_vec_1, "vgg_2_vec" : vgg_vec_2}

# Convert dict format to something jsonable
with open(out_file, 'w') as f:
    for image_id in image_points.keys():
        point_vectors = []
        for point in image_points[image_id]:
            point_dict = {}
            point_dict["xy"] = list(point)

            feature_names = image_points[image_id][point]
            for k in feature_names.keys():
                point_dict[k] = feature_names[k].tolist()

            point_vectors.append(point_dict)

        image_json = {"imageId" : image_id, "points" : point_vectors}
        print(json.dumps(image_json), file=f)

