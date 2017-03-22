#!/usr/bin/python
# Generate features for each part based on its name.

import sys
import ujson as json
import re
from ngrams import GoogleNgrams

diagram_file = sys.argv[1]
ngrams_dir = sys.argv[2]
out_file = sys.argv[3]

ngrams = GoogleNgrams(ngrams_dir)

def filter_dependencies(raw_counts, pattern):
    filtered_counts = {}
    p = re.compile(pattern)
    for (k, v) in raw_counts.iteritems():
        parts = k.split()
        for part in parts:
            result = p.match(part)
            if result is not None:
                filtered_counts[k] = v
                break
        
    return filtered_counts

type_parts = {}
with open(diagram_file, 'r') as f:
    for line in f:
        j = json.loads(line)
        diagram_label = j["label"]

        if diagram_label not in type_parts:
            type_parts[diagram_label] = set([])

        part_labels = [point["label"] for point in j["points"]]
        type_parts[diagram_label].update(part_labels)

'''
for diagram_label in type_parts.iterkeys():
    print diagram_label
    for part_label in type_parts[diagram_label]:
        print "  ", part_label
'''

# type_parts = {'tractor' : type_parts['tractor']}

all_parts = set([])
for diagram_label in type_parts.iterkeys():
    all_parts.update(type_parts[diagram_label])

print len(all_parts), "unique parts"
    
part_vectors = {}
for part in all_parts:
    query = part.split("_")[-1].strip().encode('ascii')
    print part, "->", query
    vector = ngrams.run_query(query)
    part_vectors[part] = vector

with open(out_file, 'w') as f:
    for diagram_label in type_parts.iterkeys():
        parts = type_parts[diagram_label]
        for p1 in parts:
            p1_vec = part_vectors[p1]
            for p2 in parts:
                p1_p2_counts = filter_dependencies(p1_vec, p2 + "/")
                print >> f, json.dumps( {"part1" : p1, "part2" : p2, "counts" : p1_p2_counts} )
