#!/usr/bin/python

import sys
import ujson as json
import random
import re

diagram_file = sys.argv[1]
out_file = sys.argv[2]
text_labels = ["A", "B", "C", "D", "E", "F"]    

output = []
with open(diagram_file, 'r') as f:
    j = json.load(f)

    for t in j.iterkeys():
        diagrams = j[t]
        for diagram_id in diagrams.iterkeys():
            part_labels = diagrams[diagram_id]
            label_point_map = {}

            for label in part_labels:
                label_point_map[label] = part_labels[label]

            point_annotated_id = t + "/" + diagram_id
                
            labels = sorted(label_point_map.keys())

            # shuffle the text labels for each index
            random.seed(t.__hash__())
            shuffled_text_labels = [x for x in text_labels[:len(labels)]]
            random.shuffle(shuffled_text_labels)

            points = [{"label": k, "xy" : label_point_map[k], "textId" : shuffled_text_labels[i]} for (i,k) in enumerate(labels)]
                
            width = 800
            height = 800
            output.append( {"id" : point_annotated_id, "imageId" : diagram_id, "label" : t, "points" : points, "width" : width, "height" : height} )

with open(out_file, 'w') as f:
    for d in output:
        print >> f, json.dumps(d)
