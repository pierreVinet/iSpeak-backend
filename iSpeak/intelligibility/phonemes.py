from difflib import SequenceMatcher
data = [
    ("butterfly","butterfly", True),
    ("elephant","efant", False),
    ("computer","computer", True),
    ("basketball","basball", False),
    ("telephone","tefone", False),
    ("hospital","hospital", True),
    ("umbrella","umbrela", False),
    ("sandwich","sandwich", True),
    ("home","home", True),
    ("house","house", True),
]
counts={}
for ref,trans,correct in data:
    if correct: 
        continue
    sm=SequenceMatcher(None, ref, trans)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag in ('delete','replace'):
            # characters in ref that were deleted or replaced incorrectly
            for c in ref[i1:i2]:
                counts[c]=counts.get(c,0)+1
        # we don't count insert errors because they don't correspond to missing phoneme in reference? maybe.
print(counts)