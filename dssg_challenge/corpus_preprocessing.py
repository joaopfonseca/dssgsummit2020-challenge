import os
import sys
# The ENTER key is represented as 0.
# The shift key for capitalization is represented as ^.
# The backspace key is represented as <.
# All the remaining characters not found in the valid keys are encoded as #.

with open(os.path.join(".", "data", "raw", "pt-keys.txt")) as f:
    valid_ptkeys = f.read()[:-1]

def ptpreprocessor(lines):
    def mapfun(line):
        preprocessed_line = line[:-1]
        if "<p>" in preprocessed_line:
            preprocessed_line = "0"
        elif "<" and ">" in preprocessed_line:
            preprocessed_line = None
        
        return preprocessed_line
    preprocessed_batch = list(filter(None, map(mapfun, lines)))
    preprocessed_batch = " ".join(preprocessed_batch)
    return preprocessed_batch


ptfile = os.path.join(".", "data", "raw", "CETEMPublico1.7")
ptsize = os.path.getsize(ptfile)
with open(ptfile, encoding="latin-1") as f:
    batchsize = round(ptsize / 100)
    while True:
        lines = f.readlines(batchsize)
        if not lines:
            break
        print(ptpreprocessor(lines))
        break
