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


def notebooks(root: Path, policy: dict) -> list[str]:
    import nbformat

    errors = []
    for name in policy['active_notebooks']:
        try:
            nb = nbformat.read(root / name, as_version=4)
            nbformat.validate(nb)
            cells = [c for c in nb.cells if c.cell_type == 'code']
            if not nb.cells or nb.cells[0].cell_type != 'code' or nb.cells[0].source.strip() != "from google.colab import drive\ndrive.mount('/content/drive')":
                errors.append(f'{name}: cell 0 must be the independent Drive mount')
            combined = '\n'.join(c.source for c in cells)
            if not re.search(r"REF\s*=\s*['\"][0-9a-f]{40}['\"]", combined):
                errors.append(f'{name}: missing fixed source commit')
            if 'https://github.com/' not in combined or '/content/drive/MyDrive/' not in combined:
                errors.append(f'{name}: missing GitHub source or Drive result destination')
            for cell in cells:
                ast.parse(cell.source)
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'{name}: committed execution output')
                if any(m == 'governance' or m.startswith('governance.') for m in imported_modules(cell.source, '')):
                    errors.append(f'{name}: runtime governance import')
                if re.search(r'\bRUN\w*\s*=\s*False\b|force_remount\s*=\s*True', cell.source):
                    errors.append(f'{name}: disabled run or forced remount')
        except (OSError, ValueError, SyntaxError, nbformat.ValidationError) as exc:
            errors.append(f'{name}: {exc}')
    return errors


def release_templates(root: Path, policy: dict) -> list[str]:
    import nbformat

    errors = []
    for name in policy.get('release_notebook_templates', ()):
        try:
            nb = nbformat.read(root / name, as_version=4)
            nbformat.validate(nb)
            if not nb.cells or nb.cells[0].cell_type != 'code' or nb.cells[0].source.strip() != "from google.colab import drive\ndrive.mount('/content/drive')":
                errors.append(f'{name}: cell 0 must be the independent Drive mount')
            combined = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'code')
            if 'RELEASE_SOURCE_REF = None' not in combined or 'RELEASE_SOURCE_URL = None' not in combined:
                errors.append(f'{name}: unbound release source declaration missing')
            if re.search(r"REF\s*=\s*['\"][0-9a-f]{40}['\"]", combined):
                errors.append(f'{name}: template must not claim a released source commit')
            for cell in nb.cells:
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'{name}: committed execution output')
                if cell.cell_type == 'code':
                    ast.parse(cell.source)
        except (OSError, ValueError, SyntaxError, nbformat.ValidationError) as exc:
            errors.append(f'{name}: {exc}')
    return errors


def published_release_notebooks(root: Path, policy: dict) -> list[str]:
    import nbformat

    names = policy.get('published_release_notebooks')
    if not names:
        return ['no published release notebook is declared']
    errors = []
    for name in names:
        try:
            nb = nbformat.read(root / name, as_version=4)
            nbformat.validate(nb)
            if not nb.cells or nb.cells[0].cell_type != 'code' or nb.cells[0].source.strip() != "from google.colab import drive\ndrive.mount('/content/drive')":
                errors.append(f'{name}: cell 0 must be the independent Drive mount')
            cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            combined = '\n'.join(cell.source for cell in cells)
            if not re.search(r"RELEASE_SOURCE_URL\s*=\s*['\"]https://github\.com/", combined):
                errors.append(f'{name}: missing fixed GitHub source URL')
            if not re.search(r"RELEASE_SOURCE_REF\s*=\s*['\"][0-9a-f]{40}['\"]", combined):
                errors.append(f'{name}: missing fixed published source commit')
            if 'experiments.wan_state_clock.run' not in combined or not re.search(r"experiments/wan_state_clock/configs/(fixed_terminal_validation|generate_replication)\.json", combined):
                errors.append(f'{name}: missing formal runner or state-clock config entry')
            if '/content/drive/MyDrive/' not in combined:
                errors.append(f'{name}: missing Drive result destination')
            for cell in cells:
                ast.parse(cell.source)
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'{name}: committed execution output')
                if any(module == 'governance' or module.startswith('governance.') for module in imported_modules(cell.source, '')):
                    errors.append(f'{name}: runtime governance import')
                if re.search(r'\bRUN\w*\s*=\s*False\b|force_remount\s*=\s*True', cell.source):
                    errors.append(f'{name}: disabled run or forced remount')
        except (OSError, ValueError, SyntaxError, nbformat.ValidationError) as exc:
            errors.append(f'{name}: {exc}')
    return errors


def notebook_binding(root: Path, policy: dict) -> list[str]:
    kind = policy.get('notebook_binding_kind')
    if kind == 'template':
        if not policy.get('release_notebook_templates'):
            return ['no release notebook template is declared']
        return release_templates(root, policy)
    if kind == 'published':
        return published_release_notebooks(root, policy)
    return [f'unknown notebook binding kind: {kind!r}']


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
            if config.get('protocol') != 'state_clock_v1':
                errors.append(f'{name}: unexpected baseline protocol')
        except (OSError, ValueError) as exc:
            errors.append(f'{name}: {exc}')
    return errors
