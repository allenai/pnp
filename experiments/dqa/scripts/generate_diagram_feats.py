#!/usr/bin/python
# Generate random feature vectors for each diagram part

import sys
import ujson as json
import random

diagram_label_file = sys.argv[1]
out_file = sys.argv[2]

def label_to_feature_vector(label, xy, width, height):
    DIMS = 2
    vec = [0.0] * DIMS

    vec[0] = float(xy[0]) / width
    vec[1] = float(xy[1]) / height
    return vec

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
            vec = label_to_feature_vector(label, xy, width, height)
            # print "  ", xy, label
            # print "  ", vec
            
            image_points[image_id][xy] = vec

# Convert dict format to something jsonable
image_points_json = []
for image_id in image_points.iterkeys():
    
    point_vectors = []
    for point in image_points[image_id]:
        point_dict = {}
        point_dict["xy"] = list(point)
        point_dict["vec"] = image_points[image_id][point]
        point_vectors.append(point_dict)

    image_json = {"imageId" : image_id, "points" : point_vectors}
    image_points_json.append(image_json)

with open(out_file, 'wb') as f:
    for j in image_points_json:
        print >> f, json.dumps(j)
