#!/usr/bin/python
# Generate an HTML file for visualizing the predictions of
# a matching model.

import sys
import ujson as json
import random

loss_json_file = sys.argv[1]
output_dir = sys.argv[2]

image_dir ="file:///Users/jayantk/github/pnp/data/dqa_parts_v1/"

html_header = '''
<html>
<body>
<style>
img {
  border: 1px solid black
}
.partid {
  font-size: 20;
  position: absolute;
  -webkit-transform: translateX(-50%) translateY(-50%);
  transform: translateX(-50%) translateY(-50%);
  border: solid 1px black;
  padding: 0px 2px; 
}
</style>
'''
html_footer = '''
</body></html>
'''

NUM_LABELS = 5
IMG_WIDTH=448

def image_id_to_path(imgid):
    t = imgid.split("_")[0]
    return image_dir + "/" + t + "/" + imgid

def print_loss_html(j, outfile):
    html_format = '''
<img width="%(w)s" height="%(w)s" src="%(srcpath)s"></img> <img width="%(w)s" height="%(w)s" src="%(targetpath)s"></img>
'''
    args = {'srcpath' : image_id_to_path(j["sourceImgId"]),
            'targetpath' : image_id_to_path(j["targetImgId"]),
            'w' : IMG_WIDTH}

    target_to_source = {}
    for arr in j["matching"]:
        target_to_source[arr[0]] = arr[1]

        
    source_labels = j["sourceLabel"]["partLabels"]
    print >> outfile, '<div style="position: relative">'
    for part in j["sourceParts"]:
        label = source_labels[part["ind"]]
        x = IMG_WIDTH * part["coords"]["x"] / j["sourceDims"]["x"]
        y = IMG_WIDTH * part["coords"]["y"] / j["sourceDims"]["y"]
        print >> outfile, '<p class="partid" style="background-color: lightgray; left: %s; top: %s">%s</p>' % (x, y, label)
        # Print source labels on target image
        # print >> outfile, '<p class="partid" style="background-color: lightgreen; left: %s; top: %s">%s</p>' % (x + IMG_WIDTH, y, label)

    target_labels = j["sourceLabel"]["partLabels"]
    for part in j["targetParts"]:
        x = IMG_WIDTH * part["coords"]["x"] / j["targetDims"]["x"]
        y = IMG_WIDTH * part["coords"]["y"] / j["targetDims"]["y"]
        target_ind = part["ind"]
        target_label = target_labels[target_ind]
        source_ind = target_to_source[target_ind]
        source_label = source_labels[source_ind]

        color = None
        text = None
        if source_label == target_label:
            color = "lightgreen"
            text = source_label
        else: 
            color = "red"
            text = source_label

        print >> outfile, '<p class="partid" style="background-color: %s; left: %s; top: %s">%s</p>' % (color, x + IMG_WIDTH, y, text)

    print >> outfile, html_format % args

    print >> outfile, "</div>"

def compute_confusion_matrix(jsons):
    label_list = jsons[0]["sourceLabel"]["partLabels"]
    label_inds = dict([(y, x) for (x, y) in enumerate(jsons[0]["sourceLabel"]["partLabels"])])
    num_labels = len(label_inds)
    mat = [[0 for i in xrange(num_labels)] for i in xrange(num_labels)]
    
    for j in jsons:
        source_labels = jsons[0]["sourceLabel"]["partLabels"]
        target_labels = jsons[0]["targetLabel"]["partLabels"]
        for arr in j["matching"]:
            target_ind = label_inds[target_labels[arr[0]]]
            source_ind = label_inds[source_labels[arr[1]]]
            
            mat[target_ind][source_ind] += 1

    return (mat, label_list)

def generate_confusion_matrix_html(confusion_matrix, label_list, f):
    print >> f, "<table>"
    print >> f, "<tr><td></td>"
    for j in xrange(len(confusion_matrix)):
        print >> f, "<td>", label_list[j], "</td>"
    print >> f, "<td>Accuracy:</td></tr>"
            
    for i in xrange(len(confusion_matrix)):
        print >> f, "<tr>"
        print >> f, "<td>", label_list[i], "</td>"

        for j in xrange(len(confusion_matrix[i])):
            print >> f, "<td>", confusion_matrix[i][j], "</td>"

        accuracy = 100 * float(confusion_matrix[i][i]) / sum(confusion_matrix[i])
        print >> f, "<td>%.1f%%</td>" % accuracy
        print >> f, "</tr>"
    print >> f, "</table>"


def generate_label_html(label, losses, html_output_file):
    with open(html_output_file, 'w') as f:
        print >> f, html_header
        print >> f, "<h1>" + label + "</h1>"

        print >> f, "<h2>Confusion Matrix</h2>"
        print >> f, "<p>(Rows are the true target label, columns are the predicted labels. Accuracy is % of points with the row's target label that are predicted correctly.)</p>"
        (confusion_matrix, label_list) = compute_confusion_matrix(losses)
        generate_confusion_matrix_html(confusion_matrix, label_list, f)

        for j in losses:
            print_loss_html(j, f)
        print >> f, html_footer  


losses_by_label = {}
with open(loss_json_file, 'r') as g:
    for line in g:
        j = json.loads(line)
        label_type = j["sourceImgId"].split("_")[0]
        if label_type not in losses_by_label:
            losses_by_label[label_type] = []
        losses_by_label[label_type].append(j)


for label in losses_by_label.iterkeys():
    html_output_file = output_dir + "/" + label + ".html"
    generate_label_html(label, losses_by_label[label], html_output_file)

label_accuracies = []
for label in losses_by_label.iterkeys():
    losses = losses_by_label[label]
    (confusion_matrix, label_list) = compute_confusion_matrix(losses)
    num_correct = 0
    num_total = 0
    for i in xrange(len(confusion_matrix)):
        num_correct += confusion_matrix[i][i]
        num_total += sum(confusion_matrix[i])

    accuracy = float(num_correct) / num_total
    label_accuracies.append((label, accuracy))

    label_accuracies.sort(key=lambda x: x[1])
    
    index_file = output_dir + "/index.html"
    with open(index_file, 'w') as f:
        print >> f, html_header
        for (label, acc) in label_accuracies:
            a = acc * 100
            num_examples = len(losses_by_label[label])
            print >> f, '<h3><a href="%s.html">%s</a> (%.1f%%) (%d examples)</h3>' % (label, label, acc * 100, num_examples)

            (confusion_matrix, label_list) = compute_confusion_matrix(losses_by_label[label])
            generate_confusion_matrix_html(confusion_matrix, label_list, f)

            
        print >> f, html_footer
