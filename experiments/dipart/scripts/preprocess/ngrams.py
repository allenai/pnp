import glob
import gzip
import sys
import re
from math import sqrt,log

class GoogleNgrams:

    '''
    gzip_dir is the name of a directory containing the
    gzipped parts of the ngrams data.
    '''
    def __init__(self, gzip_dir):
        self.gzip_dir = gzip_dir

        # Index the start word of each file
        self.files = glob.glob(self.gzip_dir + '/*.gz')
        self.words = []
        for f in self.files:
            start_word = self.get_start_word(f)
            self.words.append(start_word)

    def get_start_word(self, filename):
        with gzip.open(filename, 'r') as f:
            return f.readline().split('\t')[0]

    def find_files_with_word(self, query_word):
        files_to_search = []
        for i in xrange(len(self.words)):
            if query_word >= self.words[i] and (i + 1 == len(self.words) or query_word <= self.words[i + 1]):
                files_to_search.append(self.files[i])

        return files_to_search

    def run_query(self, query_word):
        filenames = self.find_files_with_word(query_word)
        results = {}
        for filename in filenames:
            q = query_word + '\t'
            with gzip.open(filename, 'r') as f:
                for line in f:
                    if line.startswith(q):
                        parts = line.split('\t')
                        results[parts[1]] = int(parts[2])

        return results
