#!/usr/bin/python

import sys
import ujson as json
import random

diagram_label_file = sys.argv[1]
out_file = sys.argv[2]

def label_to_feature_vector(label):
    DIMS = 10
    vec = [0.0] * DIMS
    h = label.__hash__() % DIMS
    vec[h] = 1.0
    return vec

image_points = {}
with open(diagram_label_file, 'r') as f:
    for line in f:
        j = json.loads(line)

        image_id = j["imageId"]

        if not image_points.has_key(image_id):
            image_points[image_id] = {}

        # print image_id
        for p in j["points"]:
            xy = tuple(p["xy"])
            label = p["label"]
            vec = label_to_feature_vector(label)
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
