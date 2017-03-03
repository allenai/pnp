#!/usr/bin/python
# Generate an HTML file for visualizing the predictions of
# a matching model.

import sys
import ujson as json
import random

loss_json_file = sys.argv[1]
html_output_file = sys.argv[2]

image_dir ="file:///Users/jayantk/github/pnp/data/dqa/all_images/"

def print_loss_html(j, outfile):
    html_format = '''
<img width="500" height="500" src="%(srcpath)s"></img> <img width="500" height="500" src="%(targetpath)s"></img>
'''
    args = {'srcpath' : image_dir + j["sourceImgId"],
            'targetpath' : image_dir + j["targetImgId"]}

    print >> outfile, '<div style="position: relative">'
    for part in j["sourceParts"]:
        x = 500 * part["coords"]["x"] / j["sourceDims"]["x"]
        y = 500 * part["coords"]["y"] / j["sourceDims"]["y"]
        print >> outfile, '<p style="color: lightgreen; position: absolute; left: %s; top: %s">%s</p>' % (x, y, part["ind"])
        print >> outfile, '<p style="color: lightgreen; position: absolute; left: %s; top: %s">%s</p>' % (x + 500, y, part["ind"])

    for part in j["targetParts"]:
        x = 500 * part["coords"]["x"] / j["targetDims"]["x"]
        y = 500 * part["coords"]["y"] / j["targetDims"]["y"]
        print >> outfile, '<p style="color: red; position: absolute; left: %s; top: %s">%s</p>' % (x + 500, y, part["ind"])

    print >> outfile, html_format % args

    print >> outfile, "</div>"

with open(html_output_file, 'w') as f:
    print >> f, "<html><body><style>img {border: 1px solid black}</style>"
    
    with open(loss_json_file, 'r') as g:
        for line in g:
            j = json.loads(line)
            print_loss_html(j, f)

    print >> f, "</body></html>"
