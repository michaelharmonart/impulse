# Impulse Rigging Utils
A set of rigging utilities for Maya.

Notable current features/code snippets that might be useful to you:
- pin.py has helper functions that allows for programatic generation of uv_pin nodes for attaching transforms to either a NURBS surface or a mesh
- pin.py has a helper function for generating a "matrixPin" which allows pinning a transform to a surface with stretch and shear
- control.py has a helper function for generating a surface_control, which is a control that moves along the surface of a given NURBS or mesh
- ribbon.py has helper functions for generating NURBS ribbons along with attached joints and controls.
- spline.py has helper functions for generating a "ribbon" like spline based on weighted matrix sums, aka it's faster and more flexible than a ribbon (allows for scaling of control verts)
- skin.py has helper funtions to automatically split weights for twist/bend joints
- color.py has a helper function to bake textures into face colors
- and many more!
