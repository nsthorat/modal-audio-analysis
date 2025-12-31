"""
Patch allin1 to work with NATTEN 0.17.5+ by adding compatibility shims.

NATTEN 0.17.5 renamed:
- natten1dqkrpb -> na1d_qk (with different arg order for rpb)
- natten1dav -> na1d_av
- natten2dqkrpb -> na2d_qk
- natten2dav -> na2d_av
"""

import os
import shutil
import site

# Find site-packages
sp = site.getsitepackages()[0]
allin1_models_dir = os.path.join(sp, "allin1", "models")
dinat_path = os.path.join(allin1_models_dir, "dinat.py")

# Remove __pycache__ to clear cached bytecode
for root, dirs, _files in os.walk(os.path.join(sp, "allin1")):
    for d in dirs:
        if d == "__pycache__":
            pycache_path = os.path.join(root, d)
            shutil.rmtree(pycache_path)
            print(f"Removed {pycache_path}")

# Read original dinat.py
with open(dinat_path) as f:
    content = f.read()

# Replace the old import with new imports and compatibility shims
old_import = "from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb"

new_import = '''# Compatibility shims for NATTEN 0.17.5+ API
from natten.functional import na1d_qk, na1d_av, na2d_qk, na2d_av

def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """Compatibility shim: old API -> new na1d_qk"""
    return na1d_qk(query, key, kernel_size, dilation, rpb=rpb)

def natten1dav(attn, value, kernel_size, dilation):
    """Compatibility shim: old API -> new na1d_av"""
    return na1d_av(attn, value, kernel_size, dilation)

def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """Compatibility shim: old API -> new na2d_qk"""
    return na2d_qk(query, key, kernel_size, dilation, rpb=rpb)

def natten2dav(attn, value, kernel_size, dilation):
    """Compatibility shim: old API -> new na2d_av"""
    return na2d_av(attn, value, kernel_size, dilation)'''

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(dinat_path, "w") as f:
        f.write(content)
    print(f"Patched {dinat_path} with NATTEN 0.17.5+ compatibility shims")
else:
    print(f"WARNING: Could not find import statement to patch in {dinat_path}")
    print("Looking for:", old_import)
