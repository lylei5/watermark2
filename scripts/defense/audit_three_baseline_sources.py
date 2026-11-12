"""Record pinned source equivalence and upstream model-weight identities on249."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    def visit_Module(self, node):
        return self._visit(node)

    def visit_ClassDef(self, node):
        return self._visit(node)

    def visit_FunctionDef(self, node):
        return self._visit(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit(node)

    def _visit(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def structure(text):
    return ast.dump(WithoutDocstrings().visit(ast.parse(text)), include_attributes=False)


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert os.name == 'posix' and root.is_dir()
    metadata = root / 'tools/baseline_metadata'
    out = root / 'runs/three_baselines_20261113'
    old_commit = 'eb7fb0ddc46a07e4f52a0a79038e546f7c925b50'
    new = root / 'third_party/frozen/PGID_dc4ad96'
    pgid = []
    for path in sorted(new.rglob('*.py')):
        rel = path.relative_to(new).as_posix()
        old = subprocess.check_output(['git', '-C', str(root / 'third_party/PGID'), 'show', old_commit + ':' + rel])
        current = path.read_bytes()
        pgid.append(dict(path=rel, previous_sha256=hashlib.sha256(old).hexdigest(), current_sha256=sha(path),
                         identical_bytes=old == current, identical_executable_ast=structure(old.decode()) == structure(current.decode())))
    assert all(x['identical_executable_ast'] for x in pgid)
    archive = metadata / 'sembind_337c9f9.tar.gz'
    official = root / 'third_party/frozen/SemBind_337c9f9'
    official.mkdir(exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            target = official.joinpath(*parts)
            assert target.resolve().is_relative_to(official.resolve())
            assert member.isdir() or member.isfile()
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    paths = json.loads((root / 'runs/sembind_chain_revalidation_20261111/paths.json').read_text())
    previous = Path(paths['official'])
    equivalence = []
    for path in sorted(official.rglob('*.py')):
        rel = path.relative_to(official).as_posix()
        active = previous / rel
        equivalence.append(dict(path=rel, pinned_sha256=sha(path), previous_present=active.exists(),
                                previous_sha256=sha(active) if active.exists() else None,
                                identical_executable_ast=structure(path.read_text()) == structure(active.read_text()) if active.exists() else None))
    preflight = json.loads((out / 'seal_preflight.json').read_text())
    weights = {}
    for key in ['blip2', 'embedding']:
        upstream = json.loads((metadata / (key + '.json')).read_text())
        rows = []
        for item in upstream['siblings']:
            if 'lfs' not in item:
                continue
            name = item['rfilename']
            actual = preflight['assets'][key]['files'].get(name)
            if actual:
                expected = item['lfs']['sha256']
                rows.append(dict(file=name, upstream_sha256=expected, local_sha256=actual['sha256'],
                                 exact=expected == actual['sha256']))
        assert rows and all(x['exact'] for x in rows), (key, rows)
        weights[key] = dict(repository=upstream['id'], revision=upstream['sha'], files=rows)
    report = dict(pgid_previous=old_commit, pgid_current='dc4ad963ffefcb4cf84340084db617472983d3e0',
                  pgid_ast=pgid, sembind_commit='337c9f90a150157ac1d410208ae20e6530ca8cdd',
                  sembind_root=str(official), sembind_archive_sha256=sha(archive),
                  sembind_files={str(p.relative_to(official)): sha(p) for p in official.rglob('*') if p.is_file()},
                  sembind_previous_path=str(previous), sembind_previous_equivalence=equivalence,
                  seal_upstream_weights=weights,
                  sd21_official_snapshot_equivalence=False,
                  sd21_note='The public stabilityai metadata request returned401; cached Manojb mirror identity is recorded separately.')
    (out / 'source_equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(pgid_python_files=len(pgid), pgid_all_executable_ast_equal=True,
                          sembind_comparison=equivalence, seal_weights=weights)))


if __name__ == '__main__':
    main()
