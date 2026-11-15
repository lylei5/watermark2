"""Freeze and execute the observable-band mechanism with paired recovery controls."""
import argparse
import contextlib
import io
from pathlib import Path
import shutil

import run_spatial_witness_recovery as driver


def prepare(run):
    with contextlib.redirect_stdout(io.StringIO()):driver.prepare(run,'visibility')
    for name in ['observable_evidence_recovery.py','run_observable_evidence.py']:
        shutil.copy2(Path(__file__).with_name(name),run/name)
    plan=driver.read(run/'protocol.json')
    plan.update(scripts={p.name:driver.sha(p) for p in run.glob('*.py')},codec_module='observable_evidence_recovery',
        variant='observability',layout='multiscale',methods=['coarse','any_scale','all_scales','observable_scales'],
        visibility='Perband median absolute selected-pair difference >=max(0.5,0.1*reference median margin);64observed symbols activate band.',
        witness_selection='Three255symbol ordinal codewords on common17x15jittered grid:Gaussian8,Gaussian4-minus8,Gaussian2-minus4;8reference-margin-selected bits per symbol.',
        registration='Fixed coordinates in all four methods; no SIFT,SSCD,CopyNCE,edit-label or attack-label input.',
        decoding='Three actualRS(255,16)decodes with shared128bit key and exact confirmation. observable_scales requires coarse and every observable detail band.',
        decoding_reuse='Four access rules reuse the same three band candidates for each authenticated record/query; report3080APIdecisions and2310band observations separately, not9240independent decodes.',
        heldout_scope='Same helper/key and evidence across four logical access rules; prospective method freeze on previously observed development images.',
        exclusions='Inactive detail bands may be omitted only by frozen unsigned margin rule. Active decoding failures cannot be relabeled erasures.',
        calibration='No attack-based fitting; prescribed absolute/relative observability floors. Genuine-only existing strong-control thresholds and descriptive matched absoluteTPR.',
        contribution_status='Mechanism test. Layered authentication and multiscale error/erasure coding are prior art; new utility/novelty not established.',
        unknown_edit_scope='Five non-enrollment families already in development data; no claim of independent unseen-edit confirmation.',
        scope='Paired existing770record-query development cases including fullImprint and basicReprompt;3080actual recovery calls. Reprompt+ scored separately after its native evaluation.')
    driver.write(run/'protocol.json',plan)
    print({'run':str(run),'protocol_sha256':driver.sha(run/'protocol.json'),'recoveries':3080},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    assert a.run.resolve().parent==driver.ROOT/'runs'
    if a.mode=='prepare':prepare(a.run)
    else:driver.execute(a.run)
