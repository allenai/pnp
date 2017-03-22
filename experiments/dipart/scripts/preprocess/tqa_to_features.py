#!/usr/bin/python
# Generate features for each part based on its name
# using positional information from the TQA dataset.

import sys
import ujson as json
import re
from collections import defaultdict

diagrams_file = sys.argv[1]
sample_file = sys.argv[2]

sample_json = None
with open(sample_file, 'r') as f:
    sample_json = json.load(f)

diagram_to_fold = {}
for (fold, diagrams) in sample_json.iteritems():
    for d in diagrams:
        diagram_to_fold[d] = fold

# Get the parts of each kind of diagram.
fold_parts = defaultdict(list)
with open(diagrams_file, 'r') as f:
    for line in f:
        j = json.loads(line)
        fold = diagram_to_fold[j["id"]]
        part_labels = [point["label"] for point in j["points"]]
        fold_parts[fold].extend(part_labels)

for (fold1, parts1) in fold_parts.iteritems():
    p1s = set(parts1)    
    for (fold2, parts2) in fold_parts.iteritems():
        p2s = set(parts2)

        inter = p1s & p2s
        fold1pct = float(len(inter)) / len(p1s)
        
        print fold1, "/", fold2, fold1pct
        for part in inter:
            print "  ", part
