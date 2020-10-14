import os
import sys
import unicodedata
from collections import defaultdict
import re
# The ENTER key is represented as 0.
# The shift key for capitalization is represented as ^.
# The backspace key is represented as <.
# All the remaining characters not found in the valid keys are encoded as #.


def deaccent(letter):
    return unicodedata.normalize('NFKD', letter).encode('ASCII', 'ignore').decode('ASCII')

# Portuguese section ---------------------------------------------------------------------------------
with open(os.path.join(".", "data", "raw", "pt-keys.txt")) as f:
    valid_ptkeys = f.read()[:-1]

ptspecialkeys_dict = defaultdict(str)
tilde_chars, circumflex_chars, accent_chars = ["Ã", "Õ"],["Â", "Ê", "Ô", "Ç"],["Á", "Í", "Ú", "Ó", "É", "À"]
for chars, replace_char in zip([tilde_chars, circumflex_chars, accent_chars], ["~", "#", "'"]):
    for char in chars:
        ptspecialkeys_dict[char] = replace_char


def ptpreprocessor(lines):
    # TODO: ADD RANDOMLY PLACED BACKSPACE CHARACTERS "<"
    # TODO: REMOVE SPACES BEFORE ".", "," AND "?"
    def mapfun(line):
        preprocessed_line = line[:-1]  # remove "\n"
        if "<p>" in preprocessed_line:
            return "0"  # ENTER key
        # if "<ext" in preprocessed_line:  # DESCOMENTAR PARA OBTER SEPARADOR
        #     return "<sep>"  # Separator
        elif "<" and ">" in preprocessed_line:
            return None  # Removing html tags
        elif preprocessed_line=="?" or preprocessed_line=="." or preprocessed_line==",":
            return preprocessed_line
        elif preprocessed_line.istitle():
            preprocessed_line = "^" + preprocessed_line  # SHIFT key
        # Upper case
        preprocessed_line = preprocessed_line.upper()
        # Putting accentuation in front of character
        preprocessed_line = "".join(
            map(lambda i: ptspecialkeys_dict[i] + deaccent(i) if i in ptspecialkeys_dict else i,
            preprocessed_line))
        # Replace unknown characters by "#"
        bad_chars = set(preprocessed_line).difference(valid_ptkeys.replace("0", ""))  # "0" number should also be encoded as "#"
        preprocessed_line = "".join(
            map(lambda i: "#" if i in bad_chars else i, preprocessed_line)
            )
        return preprocessed_line
    preprocessed_batch = " ".join(filter(None, map(mapfun, lines)))
    # preprocessed_batch = re.sub(r' .(?= )', '.', preprocessed_batch)
    return preprocessed_batch


# Creating new preprocessed corpus file
ptfile = os.path.join(".", "data", "raw", "CETEMPublico1.7")
ptoutfile = os.path.join(".", "data", "processed", "CETEMPublico1.txt")
ptsize = os.path.getsize(ptfile)
batchsize = round(ptsize / 100)

inp_file = open(ptfile, "r", encoding="latin-1")
out_file = open(ptoutfile, "w")
i = 1
while True:
    lines = inp_file.readlines(batchsize)
    if not lines:
        break
    out_file.write(ptpreprocessor(lines))
    print(f"{i}/100 iterations done.")
    i += 1
inp_file.close()
out_file.close()
    

# English section ------------------------------------------------------------------------------------
with open(os.path.join(".", "data", "raw", "en-keys.txt")) as f:
    valid_enkeys = f.read()[:-1]


def enpreprocessor(doc):
    # TODO: ADD RANDOMLY PLACED BACKSPACE CHARACTERS "<"
    # Put "^" in front of upper case words
    upper_dict = {i: "^" + i  for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    preprocessed_doc = doc.translate(str.maketrans(upper_dict))
    # Upper case
    preprocessed_doc = preprocessed_doc.upper()
    # Replace unknown characters by "#" and replace "\n" for "0"
    bad_dict = {i: "0" if i=="\n" else "#" for i in set(preprocessed_doc).difference(valid_enkeys.replace("0", ""))}
    preprocessed_doc = preprocessed_doc.translate(str.maketrans(bad_dict))
    preprocessed_doc = re.sub(r'0(?=0)', '', preprocessed_doc)  # delete "0" if it's followed by "0"
    return preprocessed_doc


# Creating new preprocessed corpus file
enfolder = os.path.join(".", "data", "raw", "bbc")
enfiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(enfolder)) for f in fn][1:]
enoutfile = os.path.join(".", "data", "processed", "bbc.txt")
out_file = open(enoutfile, "w")
for file in enfiles:
    with open(file, encoding="latin-1") as f:
        doc = f.read()
        out_file.write(enpreprocessor(doc))
out_file.close()
