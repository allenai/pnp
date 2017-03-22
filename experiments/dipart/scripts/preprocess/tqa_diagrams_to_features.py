#!/usr/bin/python
# Generate features for each part based on its name
# using positional information from the TQA dataset.

import sys
import ujson as json
import re
from collections import defaultdict

tqa_diagrams_file = sys.argv[1]
diagrams_file = sys.argv[2]

# Get the parts of each kind of diagram.
type_parts = defaultdict(set)
with open(diagrams_file, 'r') as f:
    for line in f:
        j = json.loads(line)
        diagram_label = j["label"]
        part_labels = [point["label"] for point in j["points"]]
        type_parts[diagram_label].update(part_labels)

all_parts = set([])
part_part_map = defaultdict(set)
part_counts = defaultdict(lambda: 0)
for diagram_label in type_parts.iterkeys():
    all_parts.update(type_parts[diagram_label])

    for part in type_parts[diagram_label]:
        part_part_map[part].update(type_parts[diagram_label])
        part_counts[part] += 1

sorted_counts = sorted(part_counts.items(), key=lambda x: x[1], reverse=True)

for (k,v) in sorted_counts:
    print k, v

# Read token positions from TQA
token_x = defaultdict(lambda: 0)
token_y = defaultdict(lambda: 0)
token_count = defaultdict(lambda: 0)
with open(tqa_diagrams_file, 'r') as f:
    for line in f:
        j = json.loads(line)

        for ocr in j["value"]:
            rect = ocr["rectangle"]
            text = ocr["text"]

            x = None
            y = None
            if not isinstance(rect[0], list):
                x = rect[0]
                y = rect[1]
            else:
                x = (rect[0][0] + rect[1][0]) / 2
                y = (rect[0][1] + rect[1][1]) / 2
            
            tokens = text.split()
            for token in tokens:
                # print x, y, token

                token_x[token] += x
                token_y[token] += y
                token_count[token] += 1

num_not_found = 0
for part in all_parts:
    tokens = part.split("_")
    c = 0
    x = 0
    y = 0

    for token in tokens:
        c += token_count[token]
        x += token_x[token]
        y += token_y[token]

    if c == 0:
        print part, "n/a"
        num_not_found += 1
    else:
        nx = float(x) / c
        ny = float(y) / c
        print part, nx, ny, c


print "not found: ", num_not_found, " / ", len(all_parts)
