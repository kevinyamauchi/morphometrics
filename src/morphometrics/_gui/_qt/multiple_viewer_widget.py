from copy import deepcopy
from typing import List, Optional, Tuple

import napari
import numpy as np
from magicgui import magicgui
from napari import Viewer
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer, Points, Surface
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from napari_threedee.utils.napari_utils import get_dims_displayed
from packaging.version import parse as parse_version
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ...utils.surface_utils import binary_mask_to_surface

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data

    res_layer.metadata["viewer_name"] = name

    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict={}):
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrapper(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class MultipleViewerWidget(QSplitter):
    """The main widget of the example.

    todo: add model for ortho view config
    """

    def __init__(self, viewer: napari.Viewer, side_widget: Optional[QWidget] = None):
        super().__init__()
        self.viewer = viewer

        viewer_model1 = ViewerModel(title="model1")
        viewer_model2 = ViewerModel(title="model2")
        self.ortho_viewer_models = [viewer_model1, viewer_model2]
        self._block = False

        # connect the viewer sync events
        self._connect_main_viewer_events()
        self._connect_ortho_viewer_events()

        # add the viewer widgets
        qt_viewers, viewer_splitter = self._setup_ortho_view_qt(
            self.ortho_viewer_models, viewer
        )
        self.ortho_qt_viewers = qt_viewers
        self.addWidget(viewer_splitter)

        # add a side widget if one was provided
        if side_widget is not None:
            self.addWidget(side_widget)

    def _connect_main_viewer_events(self):
        """Connect the update functions to the main viewer events.

        These events sync the ortho viewers with changes in the main viewer.
        """
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)

    def _connect_ortho_viewer_events(self):
        """Connect the update functions to the orthoviewer events.

        These events sync the main viewer with changes in the ortho viewer.
        """
        for model in self.ortho_viewer_models:
            model.dims.events.current_step.connect(self._point_update)
            model.events.status.connect(self._status_update)

    def _setup_ortho_view_qt(
        self, ortho_viewer_models: List[ViewerModel], main_viewer: Viewer
    ) -> Tuple[List[QtViewerWrapper], QSplitter]:
        # create the QtViewer objects
        qt_viewers = [
            QtViewerWrapper(main_viewer, model) for model in ortho_viewer_models
        ]

        # create and populate the QSplitter for the orthoview QtViewer
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        for qt_viewer in qt_viewers:
            viewer_splitter.addWidget(qt_viewer)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        return qt_viewers, viewer_splitter

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        for model in self.ortho_viewer_models:
            model.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            for model in self.ortho_viewer_models:
                model.layers.selection.active = None
            return

        for model in self.ortho_viewer_models:
            model.layers.selection.active = model.layers[event.value.name]

    def _point_update(self, event):
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        """Set the dims order for each of the ortho viewers.

        This is used to set the displayed dimensions in each orthview.

        todo: make configurable via ortho view config
        """
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            for model in self.ortho_viewer_models:
                model.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.ortho_viewer_models[1].dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.ortho_viewer_models[0].dims.order = order

        # order[-3:] = order[-2], order[-3], order[-1]
        # self.ortho_viewer_models[0].dims.order = order
        # order = list(self.viewer.dims.order)
        # order[-3:] = order[-1], order[-2], order[-3]
        # self.ortho_viewer_models[1].dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events.

        todo: make configurable with model
        """
        self.ortho_viewer_models[0].layers.insert(
            event.index, copy_layer(event.value, "model1")
        )
        self.ortho_viewer_models[1].layers.insert(
            event.index, copy_layer(event.value, "model2")
        )
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.ortho_viewer_models[0].layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
            self.ortho_viewer_models[1].layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)

        event.value.events.name.connect(self._sync_name)

        self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        for model in self.ortho_viewer_models:
            self.model.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        all_viewer_models = [self.viewer] + self.ortho_viewer_models
        for model in all_viewer_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        for model in self.ortho_viewer_models:
            model.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index if event.new_index < event.index else event.new_index + 1
        )
        for model in self.ortho_viewer_models:
            model.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            for model in self.ortho_viewer_models:
                setattr(
                    model.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
        finally:
            self._block = False


class MeshLabelViewerWidget(QSplitter):
    """Add a viewer of a mesh of a selected label in 3D."""

    def __init__(self, viewer: napari.Viewer, side_widget: Optional[QWidget] = None):
        super().__init__()
        self.main_viewer = viewer
        self.surface_layer = None
        self.slice_layer = None

        # make the viewer for the mesh
        self.mesh_viewer_model = ViewerModel(title="model1")

        # connect the viewer sync events
        self._connect_main_viewer_events()
        self._connect_ortho_viewer_events()

        # make the qt viewers
        qt_viewer, viewer_splitter = self._setup_ortho_view_qt(
            self.mesh_viewer_model, viewer
        )
        self.mesh_qt_viewer = qt_viewer
        self.mesh_viewer_model.dims.ndisplay = 3

        # make the label selection widget
        self.selection_widget = LabelSelectionWidget(mesh_widget=self)

        # add the widgets
        self.addWidget(self.selection_widget)
        self.addWidget(viewer_splitter)
        if side_widget is not None:
            self.addWidget(side_widget)

    def _setup_ortho_view_qt(
        self, viewer_model: List[ViewerModel], main_viewer: Viewer
    ) -> Tuple[QtViewerWrapper, QSplitter]:
        # create the QtViewer objects
        qt_viewer = QtViewerWrapper(main_viewer, viewer_model)

        # create and populate the QSplitter for the mesh QtViewer
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(qt_viewer)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        return qt_viewer, viewer_splitter

    def _connect_main_viewer_events(self):
        """Connect the update functions to the main viewer events.

        These events sync the ortho viewers with changes in the main viewer.
        """
        self.main_viewer.dims.events.current_step.connect(self._point_update)

    def _connect_ortho_viewer_events(self):
        """Connect the update functions to the orthoviewer events.

        These events sync the main viewer with changes in the ortho viewer.
        """

    def _point_update(self, event):
        """Callback from when the dims point is changed."""

    def update_surface(
        self, vertices: np.ndarray, faces: np.ndarray, values: np.ndarray
    ):
        mesh_data = (vertices, faces, values)
        if self.surface_layer is None:
            surface_layer = Surface(mesh_data)
            self.surface_layer = surface_layer

            # set up the lighting
            self.mesh_viewer_model.layers.insert(0, self.surface_layer)
            self.mesh_visual = self.mesh_qt_viewer.layer_to_visual[self.surface_layer]
            self.mesh_viewer_model.camera.events.angles.connect(self._on_camera_change)

            # setup the points layer
            self.main_points_layer = self.main_viewer.add_points(np.empty((1, 3)))
            self.mesh_points_layer = Points(np.empty((1, 3)))
            self.mesh_viewer_model.layers.insert(1, self.mesh_points_layer)

            # connect the click event and ensure the surface layer is selected
            self.surface_layer.mouse_drag_callbacks.append(self._on_mesh_clicK)
            self.mesh_viewer_model.layers.selection = {self.surface_layer}
        else:
            self.surface_layer.data = mesh_data

    def update_segment_bounding_box(self, bounding_box: np.ndarray):
        self.segment_bounding_box = bounding_box

        slice_mesh = self._make_slice_mesh()
        if self.slice_layer is None:
            self.slice_layer = Surface(slice_mesh, opacity=0.7)
            self.mesh_viewer_model.layers.insert(1, self.slice_layer)
            self.mesh_viewer_model.layers.selection = {self.surface_layer}
            self.main_viewer.dims.events.point.connect(self._on_dims_change)

        else:
            self.slice_layer.data = slice_mesh

    def _make_slice_mesh(self):
        """Make the mesh for the display the slice currently being viewed in the main viewer"""
        slice_index = self.main_viewer.dims.point[0]
        slice_mesh_vertices = np.array(
            [
                [
                    slice_index,
                    self.segment_bounding_box[0, 1],
                    self.segment_bounding_box[0, 2],
                ],
                [
                    slice_index,
                    self.segment_bounding_box[0, 1],
                    self.segment_bounding_box[1, 2],
                ],
                [
                    slice_index,
                    self.segment_bounding_box[1, 1],
                    self.segment_bounding_box[1, 2],
                ],
                [
                    slice_index,
                    self.segment_bounding_box[1, 1],
                    self.segment_bounding_box[0, 2],
                ],
            ]
        )
        slice_mesh_faces = np.array([[0, 1, 2], [0, 2, 3]])
        slice_mesh_values = np.ones((4,))
        return slice_mesh_vertices, slice_mesh_faces, slice_mesh_values

    def _on_camera_change(self, event=None):
        if self.surface_layer is None:
            return

        # get the view direction in layer coordinates
        view_direction = np.asarray(self.mesh_viewer_model.camera.view_direction)
        dims_displayed = get_dims_displayed(self.surface_layer)
        layer_view_direction = np.asarray(
            self.surface_layer._world_to_data_ray(view_direction)
        )[dims_displayed]

        # update the node
        self.mesh_visual.node.shading_filter.light_dir = -1 * layer_view_direction[::-1]

    def _on_mesh_clicK(self, layer, event):
        """Mouse callback for when clicking in on the mesh in the viewer."""
        _, triangle_index = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )

        if triangle_index is None:
            # if the click did not intersect the mesh, don't do anything
            return

        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        point = np.mean(candidate_points, axis=0)

        self.main_points_layer.data = point
        self.mesh_points_layer.data = point

    def _on_dims_change(self, event):
        mesh_data = self._make_slice_mesh()
        self.slice_layer.data = mesh_data


class LabelSelectionWidget(QWidget):
    """Widget for selecting labels to view as a mesh."""

    def __init__(
        self,
        mesh_widget: MeshLabelViewerWidget,
        labels_layer: Optional[napari.layers.Labels] = None,
    ) -> None:
        super().__init__()

        # store the widget and layer
        self.mesh_widget = mesh_widget
        self.labels_layer = labels_layer

        # create the widget to select the labels layer and label index
        self._label_selection_widget = magicgui(
            self._set_labels_layer,
            labels_layer={"choices": self._get_valid_labels_layers},
            call_button="update segment mesh",
        )

        # add widgets to layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._label_selection_widget.native)

    def _set_labels_layer(
        self, labels_layer: napari.layers.Labels, label_index: int = 1
    ):

        self._labels_layer = labels_layer
        vertices, faces, vertex_values = self._make_segment_mesh(
            label_image=labels_layer.data, label_index=label_index
        )
        self.mesh_widget.update_surface(vertices, faces, vertex_values)
        self.mesh_widget.update_segment_bounding_box(
            self._compute_bounding_box(
                label_image=labels_layer.data, label_index=label_index
            )
        )

    def _make_segment_mesh(
        self, label_image: np.ndarray, label_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make a mesh from a selected label"""
        label_mask = label_image == label_index
        mesh = binary_mask_to_surface(label_mask, n_mesh_smoothing_iterations=0)

        vertices = mesh.vertices
        faces = mesh.faces
        vertex_values = np.ones((vertices.shape[0],))

        return vertices, faces, vertex_values

    def _get_valid_labels_layers(self, combo_box) -> List[napari.layers.Labels]:
        return [
            layer
            for layer in self.mesh_widget.main_viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _compute_bounding_box(
        self, label_image: np.ndarray, label_index: int
    ) -> np.ndarray:
        """Compute the bounding box around the selected label."""
        label_mask = label_image == label_index

        segment_coordinates = np.column_stack(np.where(label_mask))
        min_coordinates = np.min(segment_coordinates, axis=0)
        max_coordinates = np.max(segment_coordinates, axis=0)

        return np.stack([min_coordinates, max_coordinates])
