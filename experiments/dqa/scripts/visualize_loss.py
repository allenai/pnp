#!/usr/bin/python
# Generate an HTML file for visualizing the predictions of
# a matching model.

import sys
import ujson as json
import random

loss_json_file = sys.argv[1]
html_output_file = sys.argv[2]

image_dir ="file:///Users/jayantk/github/pnp/data/dqa_parts_v1/"

def image_id_to_path(imgid):
    t = imgid.split("_")[0]
    return image_dir + "/" + t + "/" + imgid

def print_loss_html(j, outfile):
    html_format = '''
<img width="500" height="500" src="%(srcpath)s"></img> <img width="500" height="500" src="%(targetpath)s"></img>
'''
    args = {'srcpath' : image_id_to_path(j["sourceImgId"]),
            'targetpath' : image_id_to_path(j["targetImgId"])}

    target_to_source = {}
    for arr in j["matching"]:
        target_to_source[arr[0]] = arr[1]
    
    print >> outfile, '<div style="position: relative">'
    for part in j["sourceParts"]:
        x = 500 * part["coords"]["x"] / j["sourceDims"]["x"]
        y = 500 * part["coords"]["y"] / j["sourceDims"]["y"]
        print >> outfile, '<p class="partid" style="background-color: lightgreen; left: %s; top: %s">%s</p>' % (x, y, part["ind"])
        print >> outfile, '<p class="partid" style="background-color: lightgreen; left: %s; top: %s">%s</p>' % (x + 500, y, part["ind"])

    for part in j["targetParts"]:
        x = 500 * part["coords"]["x"] / j["targetDims"]["x"]
        y = 500 * part["coords"]["y"] / j["targetDims"]["y"]
        part_ind = part["ind"]
        source = target_to_source[part_ind]
        print >> outfile, '<p class="partid" style="background-color: red; left: %s; top: %s">%s -> %s</p>' % (x + 500, y, part_ind, source)

    print >> outfile, html_format % args

    print >> outfile, "</div>"

with open(html_output_file, 'w') as f:
    header = '''
<html>
<body>
<style>
  img {border: 1px solid black}
  .partid { font-size: 20; position: absolute; border: solid 1px black; padding: 0px 2px; }
</style>
'''
    
    print >> f, header
    
    with open(loss_json_file, 'r') as g:
        for line in g:
            j = json.loads(line)
            print_loss_html(j, f)

    print >> f, "</body></html>"
