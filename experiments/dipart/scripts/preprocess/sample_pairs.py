#!/usr/bin/python

import sys
import json
import random
import re

split_file = sys.argv[1]
out_file = sys.argv[2]
key = sys.argv[3]
samples = int(sys.argv[4])
diagram_samples = int(sys.argv[5])

def sample_pairs(diagram_list, num_per_target, num_diagrams_per_type):
    typed_diagrams = [(d, d.split('/')[0]) for d in diagram_list]
    
    diagrams_by_type = {}
    for (d, t) in typed_diagrams:
        if not t in diagrams_by_type:
            diagrams_by_type[t] = set([])
        diagrams_by_type[t].add(d)

    if num_diagrams_per_type >= 0:
        num_types_below_threshold = 0
        for t in diagrams_by_type.iterkeys():
            ds = sorted(list(diagrams_by_type[t]))
            random.seed(t.__hash__())
            random.shuffle(ds)

            if len(ds) < num_diagrams_per_type:
                num_types_below_threshold += 1

            diagrams_by_type[t] = set(ds[:num_diagrams_per_type])

        print num_types_below_threshold, "/", len(diagrams_by_type), "types below threshold of", num_diagrams_per_type

        typed_diagrams = []
        for t in diagrams_by_type.iterkeys():
            for d in diagrams_by_type[t]:
                typed_diagrams.append((d, t))

    pairs = []
    for (d, t) in typed_diagrams:
        other_diagrams = list(diagrams_by_type[t] - set([d]))
        other_diagrams.sort()

        if num_per_target >= 0:
            random.seed(d.__hash__())
            random.shuffle(other_diagrams)

            num = min(len(other_diagrams), num_per_target)
            for i in xrange(num):
                pairs.append({'src' : other_diagrams[i], 'target' : d})
        else:
            for other_diagram in other_diagrams:
                pairs.append({'src' : other_diagram, 'target' : d})

    return pairs

j = None
with open(split_file, 'r') as f:
    j = json.load(f)

pairs = sample_pairs(j[key], samples, diagram_samples)

with open(out_file, 'wb') as f:
    for pair in pairs:
        print >> f, json.dumps(pair)

