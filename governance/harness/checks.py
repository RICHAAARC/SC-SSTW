"""Small build-time checks using RPGov's profile/boundary design as reference.

Static checks do not execute notebooks, models, or scientific adjudication.
They are architecture-regression checks, not a proof of dynamic import safety.
The local implementation is intentionally compact rather than a copied RPGov
audit registry; see governance/docs/rpgov_adaptation.md.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re


def imported_modules(source: str, package: str) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package.split('.') if package else []
            if node.level:
                prefix = prefix[:len(prefix) - node.level + 1]
                base = '.'.join(prefix + ([node.module] if node.module else []))
            else:
                base = node.module or ''
            modules.append(base)
            modules.extend(base + '.' + alias.name for alias in node.names if alias.name != '*')
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            if name in ('import_module', '__import__') and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    modules.append(node.args[0].value)
    return modules


def dependencies(root: Path, policy: dict) -> list[str]:
    errors = []
    for folder in policy['code_roots']:
        for path in sorted((root / folder).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            package = '.'.join(path.relative_to(root).parts[:-1])
            try:
                modules = imported_modules(path.read_text(encoding='utf-8'), package)
            except SyntaxError as exc:
                errors.append(f'{rel}: {exc}')
                continue
            banned = ['governance']
            if folder == 'main':
                banned += ['runtime', 'experiments', 'paper_artifacts']
            if rel.startswith('runtime/wan/'):
                banned += ['runtime.c2a', 'runtime.c2t1', 'runtime.tstwv2', 'experiments']
            if rel.startswith('experiments/wan_state_clock/'):
                banned += ['runtime.c2a', 'runtime.c2t1', 'runtime.tstwv2', 'main.sc_sstw', 'experiments.feasibility']
            for module in modules:
                if any(module == b or module.startswith(b + '.') for b in banned):
                    errors.append(f'{rel}: forbidden dependency {module}')
    return errors



def published_release_notebooks(root: Path, policy: dict) -> list[str]:
    import nbformat
    names=policy.get('published_release_notebooks')
    if not names:return ['no published release notebook is declared']
    errors=[]
    for name in names:
        try:
            nb=nbformat.read(root/name,as_version=4);nbformat.validate(nb)
            if not nb.cells or nb.cells[0].cell_type!='code' or nb.cells[0].source.strip()!="from google.colab import drive\ndrive.mount('/content/drive')":
                errors.append(f'{name}: cell 0 must be the independent Drive mount')
            cells=[c for c in nb.cells if c.cell_type=='code']
            combined='\n'.join(c.source for c in cells)
            bindings=[]
            for c in cells:
                for node in ast.parse(c.source).body:
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_SHA' for t in node.targets):
                        bindings.append(ast.literal_eval(node.value))
                if c.get('outputs') or c.get('execution_count') is not None:errors.append(f'{name}: committed execution output')
                if any(m=='governance' or m.startswith('governance.') for m in imported_modules(c.source,'')):errors.append(f'{name}: runtime governance import')
                if re.search(r'\bRUN\w*\s*=\s*False\b|force_remount\s*=\s*True',c.source):errors.append(f'{name}: disabled run or forced remount')
            if len(bindings)!=1 or not isinstance(bindings[0],str) or not re.fullmatch('[0-9a-f]{40}',bindings[0]):
                errors.append(f'{name}: missing fixed published SOURCE_SHA')
            elif nb.metadata.get('candidate_binding',{}).get('source_sha')!=bindings[0]:
                errors.append(f'{name}: metadata/source binding mismatch')
            for text in ('https://github.com/RICHAAARC/SC-SSTW.git','/content/drive/MyDrive/',
                         'experiments.wan_state_clock.grow_video_reference_run',
                         'experiments/wan_state_clock/requirements-grow-video-reference.txt'):
                if text not in combined:errors.append(f'{name}: missing current reference entry {text}')
        except (OSError,ValueError,SyntaxError,nbformat.ValidationError) as exc:
            errors.append(f'{name}: {exc}')
    return errors

def notebook_binding(root: Path, policy: dict) -> list[str]:
    if policy.get('notebook_binding_kind')!='published':
        return ['current release requires a published notebook binding']
    return published_release_notebooks(root,policy)


def release(root: Path, policy: dict) -> list[str]:
    errors = [f'missing release file: {name}' for name in policy['required_files'] if not (root / name).is_file()]
    if 'dev_requirements' in policy or policy.get('required_dev_dependencies'):
        requirements = root / policy.get('dev_requirements', 'requirements-dev.txt')
        try:
            declared = {
                re.split(r'[<>=!~;\[]', line, maxsplit=1)[0].strip().lower()
                for line in requirements.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.lstrip().startswith('#')
            }
            for dependency in policy.get('required_dev_dependencies', ()):
                if dependency.lower() not in declared:
                    errors.append(f'{requirements.relative_to(root).as_posix()}: missing declared dependency {dependency}')
        except OSError as exc:
            errors.append(f'{requirements.relative_to(root).as_posix()}: {exc}')
    for name in policy.get('reproduction_entries', ()):
        try:
            ast.parse((root / name).read_text(encoding='utf-8'))
        except (OSError, SyntaxError) as exc:
            errors.append(f'{name}: {exc}')
    for name in policy['configs']:
        try:
            config = json.loads((root / name).read_text(encoding='utf-8'))
            if config.get('name') != 'grow_video_reference_v1' or config.get('fixed_denominator') != policy['fixed_denominator']:
                errors.append(f'{name}: unexpected reference identity or denominator')
        except (OSError, ValueError) as exc:
            errors.append(f'{name}: {exc}')
    if policy.get('release_manifest'):
        import hashlib
        try:
            manifest=json.loads((root/policy['release_manifest']).read_text())
            files=manifest['files']
            for name,digest in files.items():
                path=(root/name).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    errors.append(f'manifest path missing/outside release: {name}')
                elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    errors.append(f'manifest content mismatch: {name}')
            actual=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if manifest['content_sha256']!=actual:errors.append('manifest identity mismatch')
        except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(f'release manifest: {exc}')
    return errors
