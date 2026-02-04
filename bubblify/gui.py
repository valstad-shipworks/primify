"""Interactive GUI application for URDF spherization using Viser."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

from .core import EnhancedViserUrdf, Primitive, PrimitiveStore, inject_primitives_into_urdf_xml


def _rpy_to_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw to wxyz quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def _wxyz_to_rpy(wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert wxyz quaternion to roll/pitch/yaw."""
    w, x, y, z = wxyz
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return (roll, pitch, yaw)


class BubblifyApp:
    """Main application class for interactive URDF spherization."""

    def __init__(
        self,
        robot_name: str = "panda",
        urdf_path: Optional[Path] = None,
        show_collision: bool = False,
        port: int = 8080,
        spherization_yml: Optional[Path] = None,
    ):
        """Initialize the Bubblify application.

        Args:
            robot_name: Name of robot from robot_descriptions (used if urdf_path is None)
            urdf_path: Path to custom URDF file
            show_collision: Whether to show collision meshes
            port: Viser server port
            spherization_yml: Path to existing spherization YAML file to load
        """
        self.server = viser.ViserServer(port=port)
        self.show_collision = show_collision

        # Load URDF
        if urdf_path is not None:
            self.urdf = yourdfpy.URDF.load(
                str(urdf_path),  # urdf_path,
                build_scene_graph=True,
                load_meshes=True,
                build_collision_scene_graph=show_collision,
                load_collision_meshes=show_collision,
            )
            self.urdf_path = urdf_path
        else:
            self.urdf = load_robot_description(
                robot_name + "_description",
                load_meshes=True,
                build_scene_graph=True,
                load_collision_meshes=show_collision,
                build_collision_scene_graph=show_collision,
            )
            self.urdf_path = None

        # Enhanced URDF visualizer with per-link control
        self.urdf_viz = EnhancedViserUrdf(
            self.server,
            urdf_or_path=self.urdf,
            load_meshes=True,
            load_collision_meshes=show_collision,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.4),
        )

        # Primitive management
        self.primitive_store = PrimitiveStore()

        # GUI state
        self.current_primitive_id: Optional[int] = None
        self.current_link: str = ""
        self.joint_sliders: List[viser.GuiInputHandle[float]] = []
        self.transform_control: Optional[viser.TransformControlsHandle] = None
        self.radius_gizmo: Optional[viser.TransformControlsHandle] = None

        # GUI control references for syncing
        self._link_dropdown = None
        self._current_link_dropdown = None
        self._primitive_dropdown = None
        self._primitive_type_dropdown = None
        self._selected_type_text = None
        self._sphere_radius_slider = None
        self._capsule_radius_slider = None
        self._capsule_length_slider = None
        self._box_size_x_slider = None
        self._box_size_y_slider = None
        self._box_size_z_slider = None
        self._roll_slider = None
        self._pitch_slider = None
        self._yaw_slider = None
        self._primitive_color_input = None

        # Flag to prevent recursive updates
        self._updating_primitive_ui = False

        # Visibility settings
        self.show_selected_link: bool = True
        self.show_other_links: bool = True

        # Primitive opacity settings
        self.selected_sphere_opacity: float = 1.0
        self.unselected_spheres_opacity: float = 0.5
        self.other_links_spheres_opacity: float = 0.2

        # Create primitive root frame
        self.primitives_root = self.server.scene.add_frame("/primitives", show_axes=False)

        # Setup GUI
        self._setup_robot_controls()
        self._setup_visibility_controls()
        self._setup_primitive_controls()
        self._setup_export_controls()

        # Add a grid for reference
        self._add_reference_grid()

        # Initialize visibility states
        self._update_mesh_visibility()

        # Load spherization YAML if provided
        if spherization_yml is not None:
            self._load_spherization_yaml(spherization_yml)

        print(f"🎯 Bubblify server running at http://localhost:{port}")
        print("Use the GUI controls to add and edit collision primitives!")

    def _setup_robot_controls(self):
        """Setup robot configuration and visibility controls."""
        with self.server.gui.add_folder("🤖 Robot Controls"):
            # Joint sliders
            initial_config = []

            for joint_name, (lower, upper) in self.urdf_viz.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

                slider = self.server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos,
                )
                self.joint_sliders.append(slider)
                initial_config.append(initial_pos)

            # Connect sliders to URDF update
            def update_robot_config():
                config = np.array([s.value for s in self.joint_sliders])
                self.urdf_viz.update_cfg(config)

            for slider in self.joint_sliders:
                slider.on_update(lambda _: update_robot_config())

            # Apply initial configuration
            update_robot_config()

            # Reset button
            reset_joints_btn = self.server.gui.add_button("🏠 Reset to Home")

            @reset_joints_btn.on_click
            def _(_):
                for slider, init_val in zip(self.joint_sliders, initial_config):
                    slider.value = init_val

    def _setup_visibility_controls(self):
        """Setup visibility controls in separate section."""
        with self.server.gui.add_folder("👁️ Visibility Controls"):
            # Current link dropdown
            all_links = self.urdf_viz.get_all_link_names()
            if not all_links:
                all_links = ["base_link"]
            current_link_dropdown = self.server.gui.add_dropdown("Current Link", options=all_links, initial_value=all_links[0])

            # Mesh visibility toggles
            show_selected_link_cb = self.server.gui.add_checkbox("Show Selected Link", initial_value=self.show_selected_link)
            show_other_links_cb = self.server.gui.add_checkbox("Show Other Links", initial_value=self.show_other_links)

            # Primitive opacity controls with clearer names
            selected_sphere_opacity = self.server.gui.add_slider(
                "Current Primitive", min=0.0, max=1.0, step=0.1, initial_value=self.selected_sphere_opacity
            )
            unselected_spheres_opacity = self.server.gui.add_slider(
                "Other Primitives (Same Link)", min=0.0, max=1.0, step=0.1, initial_value=self.unselected_spheres_opacity
            )
            other_links_spheres_opacity = self.server.gui.add_slider(
                "Primitives (Other Links)", min=0.0, max=1.0, step=0.1, initial_value=self.other_links_spheres_opacity
            )

            # Store references for updates
            self._current_link_dropdown = current_link_dropdown

            # Set initial current link from dropdown
            self.current_link = current_link_dropdown.value

            @current_link_dropdown.on_update
            def _(_):
                self.current_link = current_link_dropdown.value
                self._sync_link_selection()
                self._update_mesh_visibility()

            @show_selected_link_cb.on_update
            def _(_):
                self.show_selected_link = show_selected_link_cb.value
                self._update_mesh_visibility()

            @show_other_links_cb.on_update
            def _(_):
                self.show_other_links = show_other_links_cb.value
                self._update_mesh_visibility()

            @selected_sphere_opacity.on_update
            def _(_):
                self.selected_sphere_opacity = selected_sphere_opacity.value
                self._update_primitive_opacities()

            @unselected_spheres_opacity.on_update
            def _(_):
                self.unselected_spheres_opacity = unselected_spheres_opacity.value
                self._update_primitive_opacities()

            @other_links_spheres_opacity.on_update
            def _(_):
                self.other_links_spheres_opacity = other_links_spheres_opacity.value
                self._update_primitive_opacities()

    def _setup_primitive_controls(self):
        """Setup simplified primitive creation and editing controls."""
        with self.server.gui.add_folder("🧩 Primitive Editor"):
            # Get links for dropdown
            all_links = self.urdf_viz.get_all_link_names()
            if not all_links:
                all_links = ["base_link"]

            # Link selection
            link_dropdown = self.server.gui.add_dropdown("Link", options=all_links, initial_value=all_links[0])
            self.current_link = link_dropdown.value
            self._link_dropdown = link_dropdown  # Store reference for syncing

            # Primitive selection dropdown (will be populated based on selected link)
            primitive_dropdown = self.server.gui.add_dropdown("Primitive", options=["None"], initial_value="None")
            self._primitive_dropdown = primitive_dropdown  # Store reference

            # Primitive creation and deletion
            primitive_type_dropdown = self.server.gui.add_dropdown(
                "New Primitive Type",
                options=["Sphere", "Cuboid", "Capsule"],
                initial_value="Sphere",
            )
            self._primitive_type_dropdown = primitive_type_dropdown

            add_primitive_btn = self.server.gui.add_button("➕ Add Primitive")
            delete_primitive_btn = self.server.gui.add_button("🗑️ Delete Selected")

            # Primitive statistics
            total_primitive_count = self.server.gui.add_text("Total Primitives", initial_value="0")
            link_primitive_count = self.server.gui.add_text("Primitives on Current Link", initial_value="0")

            # Selected type
            selected_type_text = self.server.gui.add_text("Selected Type", initial_value="None")
            self._selected_type_text = selected_type_text

            # Primitive properties
            sphere_radius = self.server.gui.add_slider("Sphere Radius", min=0.005, max=0.14, step=0.001, initial_value=0.05)
            box_x = self.server.gui.add_slider("Cuboid X", min=0.005, max=0.3, step=0.001, initial_value=0.05)
            box_y = self.server.gui.add_slider("Cuboid Y", min=0.005, max=0.3, step=0.001, initial_value=0.05)
            box_z = self.server.gui.add_slider("Cuboid Z", min=0.005, max=0.3, step=0.001, initial_value=0.05)
            capsule_radius = self.server.gui.add_slider("Capsule Radius", min=0.005, max=0.14, step=0.001, initial_value=0.03)
            capsule_length = self.server.gui.add_slider("Capsule Length", min=0.01, max=0.5, step=0.001, initial_value=0.1)

            roll_slider = self.server.gui.add_slider("Roll", min=-math.pi, max=math.pi, step=0.001, initial_value=0.0)
            pitch_slider = self.server.gui.add_slider("Pitch", min=-math.pi, max=math.pi, step=0.001, initial_value=0.0)
            yaw_slider = self.server.gui.add_slider("Yaw", min=-math.pi, max=math.pi, step=0.001, initial_value=0.0)

            primitive_color = self.server.gui.add_rgb("Color", initial_value=(255, 180, 60))

            self._sphere_radius_slider = sphere_radius
            self._box_size_x_slider = box_x
            self._box_size_y_slider = box_y
            self._box_size_z_slider = box_z
            self._capsule_radius_slider = capsule_radius
            self._capsule_length_slider = capsule_length
            self._roll_slider = roll_slider
            self._pitch_slider = pitch_slider
            self._yaw_slider = yaw_slider
            self._primitive_color_input = primitive_color

            def update_primitive_dropdown():
                """Update primitive dropdown based on selected link."""
                link_name = link_dropdown.value
                self.current_link = link_name
                primitives = self.primitive_store.get_primitives_for_link(link_name)

                if primitives:
                    options = [f"{p.shape.capitalize()} {p.id}" for p in primitives]
                    primitive_dropdown.options = options

                    primitive_to_select = None
                    if self.current_primitive_id is not None:
                        current_primitive = self.primitive_store.by_id.get(self.current_primitive_id)
                        if current_primitive and current_primitive.link == link_name:
                            primitive_to_select = current_primitive

                    if primitive_to_select is None:
                        primitive_to_select = primitives[0]

                    primitive_dropdown.value = f"{primitive_to_select.shape.capitalize()} {primitive_to_select.id}"
                    self.current_primitive_id = primitive_to_select.id
                else:
                    primitive_dropdown.options = ["None"]
                    primitive_dropdown.value = "None"
                    self.current_primitive_id = None

                self._update_transform_control()
                self._update_primitive_properties_ui()
                self._update_primitive_opacities()
                self._update_mesh_visibility()

                # Update counts
                total_primitive_count.value = str(len(self.primitive_store.by_id))
                link_primitive_count.value = str(len(primitives))

            def update_selected_primitive():
                """Update selected primitive ID from dropdown and switch link context."""
                if primitive_dropdown.value != "None":
                    primitive_id = int(primitive_dropdown.value.split()[-1])
                    self.current_primitive_id = primitive_id

                    if primitive_id in self.primitive_store.by_id:
                        primitive = self.primitive_store.by_id[primitive_id]
                        if primitive.link != self.current_link:
                            self.current_link = primitive.link
                            link_dropdown.value = primitive.link
                            self._sync_link_selection()
                            self._update_mesh_visibility()

                    self._update_transform_control()
                    self._update_primitive_properties_ui()
                    self._update_primitive_opacities()
                else:
                    self.current_primitive_id = None
                    self._remove_transform_control()

            @link_dropdown.on_update
            def _(_):
                self.current_link = link_dropdown.value
                self._sync_link_selection()
                self._update_mesh_visibility()
                update_primitive_dropdown()

            @primitive_dropdown.on_update
            def _(_):
                update_selected_primitive()

            @add_primitive_btn.on_click
            def _(_):
                """Add a new primitive to the selected link using current properties."""
                link_name = link_dropdown.value
                primitive_type = primitive_type_dropdown.value

                if primitive_type == "Sphere":
                    radius = sphere_radius.value
                    primitive = self.primitive_store.add_sphere(link_name, xyz=(0.0, 0.0, 0.0), radius=radius)
                elif primitive_type == "Cuboid":
                    size = (box_x.value, box_y.value, box_z.value)
                    primitive = self.primitive_store.add_cuboid(link_name, xyz=(0.0, 0.0, 0.0), size=size)
                else:
                    radius = capsule_radius.value
                    length = capsule_length.value
                    primitive = self.primitive_store.add_capsule(
                        link_name, xyz=(0.0, 0.0, 0.0), radius=radius, length=length
                    )

                self._create_primitive_visualization(primitive)
                self.current_primitive_id = primitive.id

                update_primitive_dropdown()
                self._update_transform_control()
                self._update_radius_gizmo()
                self._update_primitive_properties_ui()
                self._update_primitive_opacities()

            @delete_primitive_btn.on_click
            def _(_):
                """Delete the selected primitive."""
                if self.current_primitive_id is not None:
                    self.primitive_store.remove(self.current_primitive_id)
                    self.current_primitive_id = None
                    self._remove_transform_control()
                    update_primitive_dropdown()

            def update_primitive_properties():
                """Update primitive properties from UI."""
                if self._updating_primitive_ui:
                    return

                if self.current_primitive_id is None or self.current_primitive_id not in self.primitive_store.by_id:
                    return

                primitive = self.primitive_store.by_id[self.current_primitive_id]
                primitive.color = tuple(int(c) for c in primitive_color.value)

                if primitive.shape == "sphere":
                    primitive.radius = float(sphere_radius.value)
                elif primitive.shape == "cuboid":
                    primitive.size = (float(box_x.value), float(box_y.value), float(box_z.value))
                elif primitive.shape == "capsule":
                    primitive.radius = float(capsule_radius.value)
                    primitive.length = float(capsule_length.value)

                primitive.local_rpy = (float(roll_slider.value), float(pitch_slider.value), float(yaw_slider.value))

                self._update_primitive_visualization(primitive)
                self._update_radius_gizmo()

            sphere_radius.on_update(lambda _: update_primitive_properties())
            box_x.on_update(lambda _: update_primitive_properties())
            box_y.on_update(lambda _: update_primitive_properties())
            box_z.on_update(lambda _: update_primitive_properties())
            capsule_radius.on_update(lambda _: update_primitive_properties())
            capsule_length.on_update(lambda _: update_primitive_properties())
            roll_slider.on_update(lambda _: update_primitive_properties())
            pitch_slider.on_update(lambda _: update_primitive_properties())
            yaw_slider.on_update(lambda _: update_primitive_properties())
            primitive_color.on_update(lambda _: update_primitive_properties())

            # Initialize
            update_primitive_dropdown()
            self._update_primitive_opacities()

    def _setup_export_controls(self):
        """Setup export functionality."""
        with self.server.gui.add_folder("💾 Export"):
            # Get default export names based on URDF
            default_name = "spherized"
            if self.urdf_path and self.urdf_path.stem:
                default_name = f"{self.urdf_path.stem}_spherized"

            # Export name configuration (no paths, just filenames)
            export_name_input = self.server.gui.add_text("Export Name", initial_value=default_name)

            # Export options
            export_yml_btn = self.server.gui.add_button("Export Primitives (YAML)")
            export_urdf_btn = self.server.gui.add_button("Export URDF with Primitives")

            # Status with error details (read-only)
            export_status = self.server.gui.add_markdown("Ready to export")
            export_details = self.server.gui.add_markdown("")

            @export_yml_btn.on_click
            def _(_):
                """Export primitive configuration to YAML."""
                try:
                    import yaml

                    collision_primitives = {}
                    collision_spheres = {}
                    for primitive in self.primitive_store.by_id.values():
                        if primitive.link not in collision_primitives:
                            collision_primitives[primitive.link] = []
                        if primitive.link not in collision_spheres:
                            collision_spheres[primitive.link] = []

                        center = primitive.local_xyz
                        if hasattr(center, "tolist"):
                            center = center.tolist()
                        else:
                            center = [float(x) for x in center]

                        rpy = [float(v) for v in primitive.local_rpy]
                        entry = {"type": primitive.shape, "center": center, "rpy": rpy}
                        if primitive.shape == "sphere":
                            entry["radius"] = float(primitive.radius)
                            collision_spheres[primitive.link].append({"center": center, "radius": float(primitive.radius)})
                        elif primitive.shape == "cuboid":
                            entry["size"] = [float(v) for v in (primitive.size or (0.0, 0.0, 0.0))]
                        elif primitive.shape == "capsule":
                            entry["radius"] = float(primitive.radius)
                            entry["length"] = float(primitive.length)

                        collision_primitives[primitive.link].append(entry)

                    # Add metadata for import (ensure clean Python types)
                    data = {
                        "collision_primitives": collision_primitives,
                        "collision_spheres": collision_spheres,
                        "metadata": {
                            "total_primitives": int(len(self.primitive_store.by_id)),
                            "links": list(collision_primitives.keys()),
                            "export_timestamp": float(time.time()),
                        },
                    }

                    # Determine output directory (same as URDF or current working directory)
                    if self.urdf_path and self.urdf_path.parent:
                        output_dir = self.urdf_path.parent
                    else:
                        output_dir = Path.cwd()

                    output_path = output_dir / f"{export_name_input.value}.yml"
                    output_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
                    export_status.content = f"✅ Exported {len(self.primitive_store.by_id)} primitives"
                    export_details.content = f"Saved to: {output_path.name}"
                    print(f"Exported spherization to {output_path.absolute()}")

                except ImportError:
                    error_msg = "PyYAML not installed. Run: pip install PyYAML"
                    export_status.content = "❌ Missing dependency"
                    export_details.content = error_msg
                    print(f"Export failed: {error_msg}")
                except Exception as e:
                    export_status.content = f"❌ Export failed: {type(e).__name__}"
                    export_details.content = str(e)
                    print(f"Export failed: {e}")

            @export_urdf_btn.on_click
            def _(_):
                """Export URDF with collision primitives."""
                try:
                    urdf_xml = inject_primitives_into_urdf_xml(self.urdf_path, self.urdf, self.primitive_store)

                    # Determine output directory (same as URDF or current working directory)
                    if self.urdf_path and self.urdf_path.parent:
                        output_dir = self.urdf_path.parent
                    else:
                        output_dir = Path.cwd()

                    output_path = output_dir / f"{export_name_input.value}.urdf"
                    output_path.write_text(urdf_xml)
                    export_status.content = f"✅ Exported URDF with {len(self.primitive_store.by_id)} primitives"
                    export_details.content = f"Saved to: {output_path.name}"
                    print(f"Exported spherized URDF to {output_path.absolute()}")

                except Exception as e:
                    export_status.content = f"❌ URDF export failed: {type(e).__name__}"
                    export_details.content = str(e)
                    print(f"URDF export failed: {e}")

    def _create_primitive_visualization(self, primitive: Primitive):
        """Create or update the 3D visualization for a primitive."""
        # Ensure link group exists
        if primitive.link not in self.primitive_store.group_nodes:
            # Get the link frame from enhanced URDF
            link_frame = self.urdf_viz.link_frame.get(primitive.link)
            if link_frame is not None:
                self.primitive_store.group_nodes[primitive.link] = self.server.scene.add_frame(
                    f"{link_frame.name}/primitives", show_axes=False
                )
            else:
                # Fallback: create under primitives root
                self.primitive_store.group_nodes[primitive.link] = self.server.scene.add_frame(
                    f"/primitives/{primitive.link}", show_axes=False
                )

        parent_frame = self.primitive_store.group_nodes[primitive.link]

        # Create visualization with appropriate opacity
        opacity = self._get_primitive_opacity(primitive)
        if primitive.shape == "sphere":
            primitive.node = self.server.scene.add_icosphere(
                f"{parent_frame.name}/sphere_{primitive.id}",
                radius=primitive.radius,
                color=primitive.color,
                position=primitive.local_xyz,
                opacity=opacity,
                visible=True,
            )
        elif primitive.shape == "cuboid":
            size = primitive.size or (0.05, 0.05, 0.05)
            mesh = trimesh.creation.box(extents=size)
            T = trimesh.transformations.euler_matrix(
                primitive.local_rpy[0], primitive.local_rpy[1], primitive.local_rpy[2], axes="sxyz"
            )
            T[:3, 3] = primitive.local_xyz
            mesh.apply_transform(T)
            primitive.node = self.server.scene.add_mesh_simple(
                f"{parent_frame.name}/cuboid_{primitive.id}",
                mesh.vertices,
                mesh.faces,
                color=primitive.color,
                opacity=opacity,
            )
        elif primitive.shape == "capsule":
            radius = primitive.radius if primitive.radius is not None else 0.03
            length = primitive.length if primitive.length is not None else 0.1
            mesh = trimesh.creation.capsule(radius=radius, height=length)
            T = trimesh.transformations.euler_matrix(
                primitive.local_rpy[0], primitive.local_rpy[1], primitive.local_rpy[2], axes="sxyz"
            )
            T[:3, 3] = primitive.local_xyz
            mesh.apply_transform(T)
            primitive.node = self.server.scene.add_mesh_simple(
                f"{parent_frame.name}/capsule_{primitive.id}",
                mesh.vertices,
                mesh.faces,
                color=primitive.color,
                opacity=opacity,
            )
        else:
            raise ValueError(f"Unknown primitive shape: {primitive.shape}")

        # Make primitive clickable for selection
        @primitive.node.on_click
        def _(_):
            # Set primitive ID FIRST, before any other updates
            self.current_primitive_id = primitive.id
            old_link = self.current_link
            self.current_link = primitive.link

            # If we switched links, we need to update dropdowns carefully
            if old_link != self.current_link:
                self._sync_link_selection()
                # IMPORTANT: Don't call update_primitive_dropdown here as it will override our selection
                # Instead, manually update the primitive dropdown after link sync
                if self._primitive_dropdown:
                    primitives = self.primitive_store.get_primitives_for_link(self.current_link)
                    if primitives:
                        options = [f"{p.shape.capitalize()} {p.id}" for p in primitives]
                        self._primitive_dropdown.options = options
                    self._sync_primitive_selection()
            else:
                # Same link, just update primitive selection
                self._sync_primitive_selection()

            # Update visuals and UI
            self._update_transform_control()
            self._update_radius_gizmo()
            self._update_primitive_opacities()
            self._update_mesh_visibility()
            self._update_primitive_properties_ui()

    def _update_primitive_visualization(self, primitive: Primitive):
        """Update existing primitive visualization."""
        if primitive.node is not None:
            # Remove old node
            primitive.node.remove()

        # Recreate with new properties
        self._create_primitive_visualization(primitive)

    def _update_transform_control(self):
        """Update transform control for the currently selected primitive."""
        if self.current_primitive_id is not None and self.current_primitive_id in self.primitive_store.by_id:
            primitive = self.primitive_store.by_id[self.current_primitive_id]

            # Remove existing transform control
            self._remove_transform_control()

            # Get the parent frame for this primitive
            parent_frame = self.primitive_store.group_nodes.get(primitive.link)
            if parent_frame is not None:
                control_name = f"{parent_frame.name}/transform_control_{primitive.id}"

                self.transform_control = self.server.scene.add_transform_controls(
                    control_name,
                    scale=0.7,
                    disable_rotations=False,
                    position=primitive.local_xyz,
                    wxyz=_rpy_to_wxyz(*primitive.local_rpy),
                )

                # Set up callback for transform updates
                @self.transform_control.on_update
                def _(_):
                    if self.current_primitive_id is not None and self.current_primitive_id in self.primitive_store.by_id:
                        current_primitive = self.primitive_store.by_id[self.current_primitive_id]
                        current_primitive.local_xyz = tuple(self.transform_control.position)
                        current_primitive.local_rpy = _wxyz_to_rpy(tuple(self.transform_control.wxyz))
                        self._update_primitive_visualization(current_primitive)
                        self._update_radius_gizmo()
                        if self._roll_slider and self._pitch_slider and self._yaw_slider:
                            self._updating_primitive_ui = True
                            self._roll_slider.value = current_primitive.local_rpy[0]
                            self._pitch_slider.value = current_primitive.local_rpy[1]
                            self._yaw_slider.value = current_primitive.local_rpy[2]
                            self._updating_primitive_ui = False

    def _remove_transform_control(self):
        """Remove the current transform control."""
        if self.transform_control is not None:
            self.transform_control.remove()
            self.transform_control = None
        self._remove_radius_gizmo()

    def _remove_radius_gizmo(self):
        """Remove the current radius gizmo."""
        if self.radius_gizmo is not None:
            self.radius_gizmo.remove()
            self.radius_gizmo = None

    def _update_radius_gizmo(self):
        """Update radius gizmo for the currently selected sphere."""
        # Remove any previous gizmo
        self._remove_radius_gizmo()

        if self.current_primitive_id is None or self.current_primitive_id not in self.primitive_store.by_id:
            return

        s = self.primitive_store.by_id[self.current_primitive_id]
        if s.shape != "sphere":
            return

        parent_frame = self.primitive_store.group_nodes.get(s.link)
        if parent_frame is None:
            return

        # Position gizmo at 135 degrees around Z-axis for better visibility
        import math

        angle = 3 * math.pi / 4  # 135 degrees
        gizmo_pos = (
            s.local_xyz[0] + s.radius * math.cos(angle),  # X component at 45°
            s.local_xyz[1] + s.radius * math.sin(angle),  # Y component at 45°
            s.local_xyz[2],  # Same Z as center
        )

        gizmo_name = f"{parent_frame.name}/radius_gizmo_{s.id}"

        # Create rotation quaternion for 135 degrees around Z-axis
        # This rotates the gizmo's X-axis by 135 degrees, making it point diagonally
        from viser import transforms as tf

        rotation_135deg = tf.SO3.from_z_radians(angle)  # 135° rotation around Z

        # Create a single-axis gizmo that allows full bidirectional movement along the rotated X axis
        # This allows both increasing and decreasing radius, including going to zero
        self.radius_gizmo = self.server.scene.add_transform_controls(
            gizmo_name,
            scale=0.4,  # Reduce size to be less prominent
            active_axes=(True, False, False),  # Only X axis active (but now rotated)
            disable_sliders=True,
            disable_rotations=True,
            # Allow full range movement - no translation limits to enable zero radius
            wxyz=rotation_135deg.wxyz,  # Rotate the gizmo 135 degrees
            position=gizmo_pos,
        )

        @self.radius_gizmo.on_update
        def _(_):
            if self.current_primitive_id not in self.primitive_store.by_id:
                return

            s2 = self.primitive_store.by_id[self.current_primitive_id]
            gizmo_pos_current = self.radius_gizmo.position

            # Calculate new radius as distance from sphere center to gizmo position
            # This is the fundamental relationship: radius = distance from center to gizmo
            center_to_gizmo = (
                gizmo_pos_current[0] - s2.local_xyz[0],
                gizmo_pos_current[1] - s2.local_xyz[1],
                gizmo_pos_current[2] - s2.local_xyz[2],
            )
            new_radius = math.sqrt(center_to_gizmo[0] ** 2 + center_to_gizmo[1] ** 2 + center_to_gizmo[2] ** 2)
            new_radius = max(0.0, new_radius)  # Allow zero radius

            # Update sphere radius
            s2.radius = new_radius
            self._update_primitive_visualization(s2)

            # Don't reposition the gizmo here! Let the user drag it freely.
            # The gizmo position directly controls the radius - no secondary positioning logic needed.

            # Update UI slider without triggering callbacks
            if self._sphere_radius_slider:
                self._updating_primitive_ui = True
                self._sphere_radius_slider.value = new_radius
                self._updating_primitive_ui = False

    def _update_primitive_properties_ui(self):
        """Update the primitive property UI controls to reflect the currently selected primitive."""
        # Set flag to prevent recursive updates
        self._updating_primitive_ui = True

        if self.current_primitive_id is not None and self.current_primitive_id in self.primitive_store.by_id:
            primitive = self.primitive_store.by_id[self.current_primitive_id]

            if self._selected_type_text:
                self._selected_type_text.value = primitive.shape.capitalize()

            if self._sphere_radius_slider and primitive.shape == "sphere":
                self._sphere_radius_slider.value = primitive.radius
            if self._capsule_radius_slider and primitive.shape == "capsule":
                self._capsule_radius_slider.value = primitive.radius
            if self._capsule_length_slider and primitive.shape == "capsule":
                self._capsule_length_slider.value = primitive.length
            if self._box_size_x_slider and primitive.shape == "cuboid":
                self._box_size_x_slider.value = primitive.size[0] if primitive.size else 0.05
            if self._box_size_y_slider and primitive.shape == "cuboid":
                self._box_size_y_slider.value = primitive.size[1] if primitive.size else 0.05
            if self._box_size_z_slider and primitive.shape == "cuboid":
                self._box_size_z_slider.value = primitive.size[2] if primitive.size else 0.05

            if self._roll_slider:
                self._roll_slider.value = primitive.local_rpy[0]
            if self._pitch_slider:
                self._pitch_slider.value = primitive.local_rpy[1]
            if self._yaw_slider:
                self._yaw_slider.value = primitive.local_rpy[2]

            if self._primitive_color_input:
                self._primitive_color_input.value = primitive.color
        else:
            # Reset to default values when no primitive selected
            if self._selected_type_text:
                self._selected_type_text.value = "None"
            if self._sphere_radius_slider:
                self._sphere_radius_slider.value = 0.05
            if self._capsule_radius_slider:
                self._capsule_radius_slider.value = 0.03
            if self._capsule_length_slider:
                self._capsule_length_slider.value = 0.1
            if self._box_size_x_slider:
                self._box_size_x_slider.value = 0.05
            if self._box_size_y_slider:
                self._box_size_y_slider.value = 0.05
            if self._box_size_z_slider:
                self._box_size_z_slider.value = 0.05
            if self._roll_slider:
                self._roll_slider.value = 0.0
            if self._pitch_slider:
                self._pitch_slider.value = 0.0
            if self._yaw_slider:
                self._yaw_slider.value = 0.0
            if self._primitive_color_input:
                self._primitive_color_input.value = (255, 180, 60)

        # Clear flag after UI update
        self._updating_primitive_ui = False

    def _sync_link_selection(self):
        """Sync link selection between visibility controls and primitive editor."""
        # Sync visibility dropdown if different
        if self._current_link_dropdown and self._current_link_dropdown.value != self.current_link:
            self._current_link_dropdown.value = self.current_link
        # Sync primitive editor dropdown if different
        if self._link_dropdown and self._link_dropdown.value != self.current_link:
            self._link_dropdown.value = self.current_link

    def _sync_primitive_selection(self):
        """Sync primitive dropdown to reflect the currently selected primitive."""
        if self._primitive_dropdown and self.current_primitive_id is not None:
            # Find the correct dropdown option for this primitive
            if self.current_primitive_id in self.primitive_store.by_id:
                primitive = self.primitive_store.by_id[self.current_primitive_id]
                expected_value = f"{primitive.shape.capitalize()} {primitive.id}"

                # Check if this value exists in the dropdown options
                if expected_value in self._primitive_dropdown.options:
                    self._primitive_dropdown.value = expected_value

    def _get_primitive_opacity(self, primitive: Primitive) -> float:
        """Get the appropriate opacity for a primitive based on current selection state."""
        if primitive.id == self.current_primitive_id:
            return self.selected_sphere_opacity
        elif primitive.link == self.current_link:
            return self.unselected_spheres_opacity
        else:
            return self.other_links_spheres_opacity

    def _update_mesh_visibility(self):
        """Update visibility of robot meshes based on link selection."""
        for link_name, mesh_handles in self.urdf_viz.link_meshes.items():
            for mesh_handle in mesh_handles:
                # Determine if this link should be visible
                if link_name == self.current_link:
                    # This is the selected link
                    mesh_handle.visible = self.show_selected_link
                else:
                    # This is a non-selected link
                    mesh_handle.visible = self.show_other_links

    def _update_primitive_opacities(self):
        """Update opacity of all primitives based on current selection state."""
        for primitive in self.primitive_store.by_id.values():
            if primitive.node is not None:
                new_opacity = self._get_primitive_opacity(primitive)
                # Update primitive opacity
                primitive.node.opacity = new_opacity
                # Handle visibility (0.0 opacity = invisible)
                primitive.node.visible = new_opacity > 0.0

    def _load_spherization_yaml(self, yaml_path: Path):
        """Load primitive configuration from YAML file at startup."""
        try:
            import yaml

            if not yaml_path.exists():
                print(f"⚠️  Spherization YAML file not found: {yaml_path}")
                return

            print(f"📥 Loading spherization from: {yaml_path}")
            data = yaml.safe_load(yaml_path.read_text())
            collision_primitives = data.get("collision_primitives", {})
            collision_spheres = data.get("collision_spheres", {})

            # Import primitives
            total_loaded = 0
            for link_name, primitives_data in collision_primitives.items():
                for primitive_data in primitives_data:
                    ptype = primitive_data.get("type", "sphere")
                    center = tuple(primitive_data.get("center", (0.0, 0.0, 0.0)))
                    rpy = tuple(primitive_data.get("rpy", (0.0, 0.0, 0.0)))
                    if ptype == "sphere":
                        radius = float(primitive_data.get("radius", 0.05))
                        primitive = self.primitive_store.add_sphere(link_name, xyz=center, radius=radius)
                    elif ptype == "cuboid":
                        size = tuple(primitive_data.get("size", (0.05, 0.05, 0.05)))
                        primitive = self.primitive_store.add_cuboid(link_name, xyz=center, size=size)
                    elif ptype == "capsule":
                        radius = float(primitive_data.get("radius", 0.03))
                        length = float(primitive_data.get("length", 0.1))
                        primitive = self.primitive_store.add_capsule(link_name, xyz=center, radius=radius, length=length)
                    else:
                        continue

                    primitive.local_rpy = rpy
                    self._create_primitive_visualization(primitive)
                    total_loaded += 1

            # Import legacy spheres if no primitives section exists
            if not collision_primitives:
                for link_name, spheres_data in collision_spheres.items():
                    for sphere_data in spheres_data:
                        sphere = self.primitive_store.add_sphere(
                            link_name, xyz=tuple(sphere_data["center"]), radius=sphere_data["radius"]
                        )
                        self._create_primitive_visualization(sphere)
                        total_loaded += 1

            print(f"✅ Loaded {total_loaded} primitives from {yaml_path.name}")

        except ImportError:
            print("⚠️  PyYAML not installed. Cannot load spherization YAML.")
            print("   Install with: pip install PyYAML")
        except Exception as e:
            print(f"❌ Failed to load spherization YAML: {e}")

    def _add_reference_grid(self):
        """Add a reference grid to the scene."""
        # Get scene bounds to position grid appropriately
        try:
            trimesh_scene = self.urdf_viz._urdf.scene or self.urdf_viz._urdf.collision_scene
            z_pos = trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0
        except:
            z_pos = 0.0

        self.server.scene.add_grid(
            "/reference_grid",
            width=2,
            height=2,
            position=(0.0, 0.0, z_pos),
            cell_color=(200, 200, 200),
            cell_thickness=1.0,
        )

    def run(self):
        """Run the application (blocking)."""
        print("🚀 Application running! Use Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down Bubblify...")
        finally:
            # Cleanup
            self._remove_transform_control()
            self._remove_radius_gizmo()
            self.urdf_viz.remove()
