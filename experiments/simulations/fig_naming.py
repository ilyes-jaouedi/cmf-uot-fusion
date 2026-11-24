"""Legend names of the two CMF-UOT variants, which differ between documents.

Code:   gcd_uot1 = greedy_coordinate_descent        (b = P 1, optimise P only)
        gcd_uot2 = greedy_block_coordinate_descent  (b decoupled from P)

paper:  extension/paper/Journal_Draft.tex calls the decoupled variant CMF-UOT1
thesis: Manuscript chapter 3 calls the b = P 1 variant CMF-UOT1

Figures are saved once per naming: paper names in OUTPUT_DIR, thesis names in
OUTPUT_DIR/thesis.
"""
import os

LABELS = {
    'paper':  {'uot1': 'CMF-UOT2', 'uot2': 'CMF-UOT1'},
    'thesis': {'uot1': 'CMF-UOT1', 'uot2': 'CMF-UOT2'},
}
NAMINGS = ('paper',)


def out_dir(base, naming):
    d = base if naming == 'paper' else os.path.join(base, 'thesis')
    os.makedirs(d, exist_ok=True)
    return d
