import bpy
import mathutils

body = bpy.data.objects.get('Auto2')
if not body:
    print("Error: Auto2 object not found")
else:
    print(f"Working on Auto2 - Current dimensions: {body.dimensions.x:.4f} x {body.dimensions.y:.4f} x {body.dimensions.z:.4f}")

    # Enter edit mode
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = body.data

    # Find the Y boundaries (left/right wheel arches stick out in Y direction)
    # Target width should be around 0.149m (14.9cm)
    target_half_width = 0.149 / 2  # 7.45cm from center

    vertices_to_clamp = []
    for i, vert in enumerate(mesh.vertices):
        world_pos = body.matrix_world @ vert.co
        # If vertex is too far out in +Y or -Y direction, clamp it
        if abs(world_pos.y) > target_half_width:
            vertices_to_clamp.append((i, world_pos))

    print(f"Found {len(vertices_to_clamp)} vertices extending beyond target width")

    if vertices_to_clamp:
        for i, world_pos in vertices_to_clamp:
            # Clamp Y coordinate to target width
            new_y = target_half_width if world_pos.y > 0 else -target_half_width
            new_world_pos = mathutils.Vector((world_pos.x, new_y, world_pos.z))
            # Convert back to local space
            local_pos = body.matrix_world.inverted() @ new_world_pos
            mesh.vertices[i].co = local_pos

        mesh.update()
        print(f"Clamped {len(vertices_to_clamp)} vertices to target width")
        print(f"New dimensions: {body.dimensions.x:.4f} x {body.dimensions.y:.4f} x {body.dimensions.z:.4f}")
    else:
        print("No vertices need trimming")

    print("\nWheel arches trimmed successfully!")
