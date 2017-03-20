#!/usr/bin/python
# Generate features for each part based on its name.

import sys
import ujson as json
import re

ngrams_file = sys.argv[1]
# out_file = sys.argv[2]

def counts_to_features(counts):
    prep_pattern = re.compile("([^ ]*)/[^ ]*/prep/.*")
    prep_counts = {}
    for (k, v) in counts.iteritems():
        m = prep_pattern.search(k)
        if m is not None:
            prep = m.group(1)
            if prep not in prep_counts:
                prep_counts[prep] = 0
            prep_counts[prep] += v

    return prep_counts


ngram_features = {}
with open(ngrams_file, 'r') as f:
    for line in f:
        j = json.loads(line)
        features = counts_to_features(j["counts"])

        if j["part1"] != j["part2"]:
            print j["part1"], j["part2"]

            for (k, v) in features.iteritems():
                if "/CC/" in k:
                    continue
                
                print "  ", k, v

        ngram_features[(j["part1"], j["part2"])] = features

all_features = set([])
for (k, counts) in ngram_features.iteritems():
    all_features.update(counts.keys())

feature_indexes = dict([(f, i) for (i, f) in enumerate(all_features)])

