#!/usr/bin/env python3
"""
Parse VoxelCAD user docs and generate a rendering script for code examples.

Usage:
    python docs/tools/extract_examples.py

Reads docs/user/*.md, extracts Python code blocks that create VoxelCAD
models, and generates docs/tools/render_examples.py which renders each
example to PNG via PyVista offscreen.

Output images: docs/user/_images/{doc_stem}/{heading_slug}_{idx}_{var}.png

Discovery: VoxelCAD model classes and transform methods are discovered
from the voxelcad package at runtime, not hardcoded.
"""

import ast
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent


def _discover_voxelcad():
    """Introspect voxelcad to find model classes and transform methods."""
    # Add source to path so we can import
    src = Path(__file__).resolve().parent.parent.parent / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from voxelcad.voxel_model import VoxelModel
    import voxelcad

    # Find all VoxelModel subclasses accessible from the package
    model_classes = set()
    for name in dir(voxelcad):
        obj = getattr(voxelcad, name)
        if (inspect.isclass(obj)
                and issubclass(obj, VoxelModel)
                and obj is not VoxelModel):
            model_classes.add(name)

    # Also check gyroid_cube for non-exported subclasses
    from voxelcad import gyroid_cube
    for name in dir(gyroid_cube):
        obj = getattr(gyroid_cube, name)
        if (inspect.isclass(obj)
                and issubclass(obj, VoxelModel)
                and obj is not VoxelModel):
            model_classes.add(name)

    # Find transform methods on VoxelModel that return TransformedModel
    transform_methods = set()
    for name, method in inspect.getmembers(VoxelModel, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue
        # Check if the method is a transform by looking for known patterns
        src_lines = inspect.getsource(method)
        if 'TransformedModel' in src_lines or 'apply_transformation' in src_lines:
            transform_methods.add(name)

    return model_classes, transform_methods


# Discover at import time
MODEL_CLASSES, TRANSFORM_METHODS = _discover_voxelcad()


@dataclass
class Example:
    doc_stem: str
    heading: str
    heading_slug: str
    block_idx: int
    code: str
    model_vars: list
    line_number: int


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text)
    return text.strip('-')


def _is_constructor_call(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in MODEL_CLASSES)


def _is_model_expr(node, known):
    if _is_constructor_call(node):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in TRANSFORM_METHODS:
            return True
        if isinstance(node.func.value, ast.Call):
            return _is_model_expr(node.func.value, known)
        if isinstance(node.func.value, ast.Name) and node.func.value.id in known:
            return True
    if isinstance(node, ast.BinOp):
        left_ok = (_is_model_expr(node.left, known)
                   or (isinstance(node.left, ast.Name) and node.left.id in known))
        right_ok = (_is_model_expr(node.right, known)
                    or (isinstance(node.right, ast.Name) and node.right.id in known))
        return left_ok or right_ok
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        if isinstance(node.operand, ast.Name) and node.operand.id in known:
            return True
    return False


def find_model_variables(code_text):
    """Use AST to find variables that hold VoxelModel instances."""
    clean = re.sub(r'\.\s*plot\s*\([^)]*\)', '', code_text)
    clean = re.sub(r'\.\s*export\s*\([^)]*\)', '', clean)
    clean = re.sub(r'^\s*print\s*\(.*\)\s*$', 'pass', clean, flags=re.MULTILINE)

    try:
        tree = ast.parse(clean)
    except SyntaxError:
        return []

    all_models = set()
    constructor_vars = []
    derived_vars = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if _is_constructor_call(node.value):
                all_models.add(name)
                if name not in constructor_vars:
                    constructor_vars.append(name)
            elif _is_model_expr(node.value, all_models):
                all_models.add(name)
                if name not in derived_vars:
                    derived_vars.append(name)

    return derived_vars if derived_vars else constructor_vars


def _is_voxelcad_code(code_text):
    if re.search(r'from\s+voxelcad\s+import', code_text):
        return True
    for cls in MODEL_CLASSES:
        if re.search(r'\b' + cls + r'\s*\(', code_text):
            return True
    return False


def parse_markdown(filepath):
    examples = []
    doc_stem = filepath.stem
    lines = filepath.read_text().splitlines()

    current_heading = 'top'
    current_slug = 'top'
    block_counts = {}

    in_code = False
    code_lines = []
    code_start = 0

    for i, line in enumerate(lines, 1):
        hm = re.match(r'^(#{1,4})\s+(.+)$', line)
        if hm and not in_code:
            current_heading = hm.group(2).strip()
            current_slug = slugify(current_heading)
            continue

        if line.strip().startswith('```') and not in_code:
            lang = line.strip()[3:].strip().lower()
            if lang in ('python', 'py'):
                in_code = True
                code_lines = []
                code_start = i
            continue

        if line.strip() == '```' and in_code:
            in_code = False
            code_text = '\n'.join(code_lines)

            if _is_voxelcad_code(code_text):
                mvars = find_model_variables(code_text)
                if mvars:
                    idx = block_counts.get(current_slug, 0)
                    block_counts[current_slug] = idx + 1
                    examples.append(Example(
                        doc_stem=doc_stem,
                        heading=current_heading,
                        heading_slug=current_slug,
                        block_idx=idx,
                        code=code_text,
                        model_vars=mvars,
                        line_number=code_start,
                    ))
            continue

        if in_code:
            code_lines.append(line)

    return examples


def _clean_code(code):
    """Strip .plot(), .export(), print() for execution in render script."""
    c = re.sub(r'^\s*\S+\.plot\([^)]*\)\s*$', '', code, flags=re.MULTILINE)
    c = re.sub(r'^\s*\S+\.export\([^)]*\)\s*$', '', code, flags=re.MULTILINE)
    c = re.sub(r'\.plot\([^)]*\)', '', c)
    c = re.sub(r'\.export\([^)]*\)', '', c)
    c = re.sub(r'^\s*print\(.*\)\s*$', '', c, flags=re.MULTILINE)
    c = re.sub(r'\n{3,}', '\n\n', c)
    return c.strip()


def _build_imports():
    """Build import block from discovered classes."""
    # Classes exported from voxelcad.__init__
    import voxelcad
    exported = []
    for name in sorted(MODEL_CLASSES):
        if hasattr(voxelcad, name):
            exported.append(name)

    # Classes only in submodules
    submodule = [n for n in sorted(MODEL_CLASSES) if n not in exported]

    lines = []
    if exported:
        lines.append(f"from voxelcad import {', '.join(exported)}")
    if submodule:
        lines.append(f"from voxelcad.gyroid_cube import {', '.join(submodule)}")
    lines.append("import voxelcad.environment as ENV")
    return '\n'.join(lines)


def generate_script(examples):
    imports = _build_imports()

    header = f'''#!/usr/bin/env python3
"""
Auto-generated by extract_examples.py -- do not edit manually.
Regenerate: python docs/tools/extract_examples.py

Renders VoxelCAD doc examples to PNG via PyVista offscreen.

Usage:
    python docs/tools/render_examples.py [--dry-run] [--filter PATTERN]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('VTK_SILENCE_GET_VOID_POINTER_WARNINGS', '1')

import pyvista as pv
pv.OFF_SCREEN = True

{imports}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = REPO_ROOT / 'docs' / 'user' / '_images'


def render_model(model, output_path, window_size=(800, 600)):
    """Render a VoxelModel to PNG."""
    mesh = model.render_volume_mesh()
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color='steelblue', show_edges=False)
    pl.camera_position = 'iso'
    pl.set_background('white')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(output_path))
    pl.close()

'''

    functions = []
    registry = []

    for ex in examples:
        safe_doc = ex.doc_stem.replace('-', '_')
        safe_head = ex.heading_slug.replace('-', '_')
        func_name = f'render_{safe_doc}__{safe_head}__{ex.block_idx}'

        cleaned = _clean_code(ex.code)
        code_body = indent(cleaned, '    ')

        render_lines = []
        for var in ex.model_vars:
            img = f"{ex.heading_slug}_{ex.block_idx}_{var}.png"
            render_lines.append(
                f"    render_model({var}, IMAGES_DIR / '{ex.doc_stem}' / '{img}')"
            )
        renders = '\n'.join(render_lines)

        func = f'''
def {func_name}():
    # Source: docs/user/{ex.doc_stem}.md
    # Section: {ex.heading} (line {ex.line_number})
    # Models: {', '.join(ex.model_vars)}
{code_body}

{renders}
'''
        functions.append(func)

        label = f'{ex.doc_stem}/{ex.heading_slug}_{ex.block_idx}'
        registry.append(f"    ('{label}', {func_name}),")

    reg_block = 'EXAMPLES = [\n' + '\n'.join(registry) + '\n]\n'

    main = '''

def main():
    parser = argparse.ArgumentParser(description='Render VoxelCAD doc examples')
    parser.add_argument('--dry-run', action='store_true',
                        help='List examples without rendering')
    parser.add_argument('--filter', type=str, default='',
                        help='Only render examples matching pattern')
    args = parser.parse_args()

    selected = [(label, fn) for label, fn in EXAMPLES if args.filter in label]

    if args.dry_run:
        for label, _ in selected:
            print(f'  {label}')
        print(f'\\n{len(selected)} examples would be rendered.')
        print(f'Output: {IMAGES_DIR}/')
        return

    print(f'Rendering {len(selected)} examples to {IMAGES_DIR}/')
    failed = []
    for i, (label, fn) in enumerate(selected, 1):
        print(f'  [{i}/{len(selected)}] {label} ... ', end='', flush=True)
        t0 = time.time()
        try:
            fn()
            print(f'ok ({time.time() - t0:.1f}s)')
        except Exception as e:
            print(f'FAILED: {e}')
            failed.append((label, str(e)))

    print()
    if failed:
        print(f'{len(failed)} failures:')
        for label, err in failed:
            print(f'  {label}: {err}')
        sys.exit(1)
    else:
        print(f'All {len(selected)} examples rendered.')


if __name__ == '__main__':
    main()
'''

    return header + '\n'.join(functions) + '\n\n' + reg_block + main


def main():
    print(f'Discovered {len(MODEL_CLASSES)} model classes: {sorted(MODEL_CLASSES)}')
    print(f'Discovered {len(TRANSFORM_METHODS)} transform methods: {sorted(TRANSFORM_METHODS)}')
    print()

    repo_root = Path(__file__).resolve().parent.parent.parent
    docs_dir = repo_root / 'docs' / 'user'
    output_path = Path(__file__).resolve().parent / 'render_examples.py'

    if not docs_dir.exists():
        print(f'Error: {docs_dir} not found', file=sys.stderr)
        sys.exit(1)

    all_examples = []
    for md_file in sorted(docs_dir.glob('*.md')):
        examples = parse_markdown(md_file)
        if examples:
            print(f'  {md_file.name}: {len(examples)} renderable examples')
            all_examples.extend(examples)

    if not all_examples:
        print('No renderable VoxelCAD examples found.')
        sys.exit(0)

    script = generate_script(all_examples)
    output_path.write_text(script)
    output_path.chmod(0o755)

    n_docs = len(set(e.doc_stem for e in all_examples))
    n_images = sum(len(e.model_vars) for e in all_examples)
    print(f'\nGenerated {output_path.name}')
    print(f'  {len(all_examples)} examples from {n_docs} docs -> {n_images} images')
    print(f'\nTo render:  python {output_path}')
    print(f'To preview: python {output_path} --dry-run')


if __name__ == '__main__':
    main()
