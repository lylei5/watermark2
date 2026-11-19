"""Unconditioned holdout plus a separate preselected low-color prompt stratum."""
import re
import sys

from color_joint_sources import ROOT, select
from color_veto_sources import schedule, run_shard

RUN=ROOT/'runs/color_richness_20261120'
GRAY_STYLE=re.compile(r'\b(?:black[ -]+and[ -]+white|b&w|monochrom\w*|gr[ae]yscale|pencil sketch|charcoal drawing|line art|ink drawing)\b',re.I)


if __name__=='__main__':
    if len(sys.argv)>1:
        run_shard(RUN,*map(int,sys.argv[1:]))
    else:
        if not (RUN/'sources.json').exists():
            strata=[(d,'random',30,None) for d in ('COCO','SDP')]
            strata.extend((d,'enriched',20,GRAY_STYLE) for d in ('COCO','SDP'))
            select(RUN,[ROOT/f'runs/{s}/sources.json' for s in ('color_joint_20261119','color_veto_20261119')],
                   'color-richness-order-v1','color-richness-seeds-v1',strata)
        schedule(RUN,__file__)
