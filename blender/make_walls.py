import bpy
import bmesh
from mathutils import Vector

def find_boundary_loops(bm):
    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    loops = []
    while boundary_edges:
        loop = []
        edge = boundary_edges.pop(0)
        v_start, v_next = edge.verts[0], edge.verts[1]
        loop.append(v_start)
        current = v_next
        prev = v_start
        while True:
            loop.append(current)
            next_edges = [e for e in current.link_edges if e in boundary_edges and prev not in e.verts]
            if not next_edges:
                break
            next_edge = next_edges[0]
            boundary_edges.remove(next_edge)
            prev, current = current, next_edge.other_vert(current)
            if current == loop[0]:
                break
        loops.append(loop)
    return loops

def extrude_outline_skip_small_gaps(obj, height=100, min_gap=2.0):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    
    loops = find_boundary_loops(bm)
    if not loops:
        print("No boundary loops found!")
        bpy.ops.object.mode_set(mode='OBJECT')
        return
    
    for outline_verts in loops:
        n = len(outline_verts)
        # Prepare a list for new verts, but only create if gap is large
        new_verts = [None] * n
        for i in range(n):
            v1 = outline_verts[i]
            v2 = outline_verts[(i+1)%n]
            if (v1.co - v2.co).length >= min_gap:
                # Create new vertex above v1 if not already created
                if new_verts[i] is None:
                    new_co = v1.co.copy()
                    new_co.z += height
                    new_verts[i] = bm.verts.new(new_co)
                # Create new vertex above v2 if not already created
                if new_verts[(i+1)%n] is None:
                    new_co2 = v2.co.copy()
                    new_co2.z += height
                    new_verts[(i+1)%n] = bm.verts.new(new_co2)
                # Create wall face
                bm.faces.new([v1, v2, new_verts[(i+1)%n], new_verts[i]])
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
    
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Outline extruded, skipping small gaps!")

# Usage:
obj = bpy.context.active_object
extrude_outline_skip_small_gaps(obj, height=100, min_gap=0.5)  # Adjust min_gap as needed