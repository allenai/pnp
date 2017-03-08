#!/usr/bin/python

import sys
import ujson as json
import random
import re

diagram_file = sys.argv[1]
diagram_size_file = sys.argv[2]
out_file = sys.argv[3]
text_labels = ["A", "B", "C", "D", "E", "F"]

diagram_sizes = {}
with open(diagram_size_file, 'r') as f:
    lines = f.readlines()

    for ind in xrange(len(lines) / 3):
        i = ind * 3
        idline = lines[i]
        id = idline[(idline.rfind('/') + 1):].strip()
        height = re.search("[0-9]+", lines[i + 1].strip()).group(0)
        width = re.search("[0-9]+", lines[i + 2].strip()).group(0)

        # print id, width, height
        diagram_sizes[id] = (int(width), int(height))
    

output = []
with open(diagram_file, 'r') as f:
    j = json.load(f)

    for t in j.iterkeys():
        diagrams = j[t]
        for diagram_id in diagrams.iterkeys():
            if not diagram_id in diagram_sizes:
                print "WARNING: could not find size for", diagram_id, "type:", t
                continue
            
            part_labels = diagrams[diagram_id]
            label_point_map = dict([(x, []) for x in part_labels])

            for label in part_labels:
                for annotation_id in part_labels[label]:
                    label_point_map[label].extend(part_labels[label][annotation_id]["annotation"])

            for annotation_ind in xrange(3):
                point_annotated_id = diagram_id + "_" + unicode(annotation_ind)

                labels = sorted(label_point_map.keys())

                # shuffle the text labels for each index
                random.seed(annotation_ind + t.__hash__())
                shuffled_text_labels = [x for x in text_labels[:len(labels)]]
                random.shuffle(shuffled_text_labels)

                points = [{"label": k, "xy" : label_point_map[k][annotation_ind], "textId" : shuffled_text_labels[i]} for (i,k) in enumerate(labels)]
                
                (width, height) = diagram_sizes[diagram_id]
                output.append( {"id" : point_annotated_id, "imageId" : diagram_id, "label" : t, "points" : points, "width" : width, "height" : height} )


with open(out_file, 'w') as f:
    for d in output:
        print >> f, json.dumps(d)
