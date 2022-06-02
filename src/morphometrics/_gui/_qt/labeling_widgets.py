from typing import List, Optional

import napari
from magicgui import magicgui, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ...label.image_utils import expand_selected_labels


class QtLabelingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._layer = None
        self._curating = False

        # create the label selection widget
        self._label_selection_widget = magicgui(
            self._toggle_curating,
            labels_layer={"choices": self._get_valid_labels_layers},
            call_button="start curating",
        )

        # create the label expansion widget
        self._label_expansion_widget = magicgui(
            self._expand_selected_labels_widget_function,
            pbar={"visible": False, "max": 0, "label": "working..."},
            call_button="expand labels",
        )
        self._label_expansion_widget.native.setVisible(False)

        # add widgets to layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._label_selection_widget.native)
        self.layout().addWidget(self._label_expansion_widget.native)

    @property
    def layer(self) -> Optional[napari.layers.Labels]:
        """The selected labels layer for curation.

        Returns
        -------
        layer : Optional[napari.layers.Labels]
            The selected labels layer that will be curated. If no layer is selected,
            returns None.
        """
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[napari.layers.Labels]):
        """The selected labels layer for curation.

        Parameters
        ----------
        layer : Optional[napari.layers.Labels]
            The selected labels layer that will be curated. If no layer is selected,
            set to None.
        """
        if layer == self._layer:
            # if the layer hasn't changed, don't perform the update
            return None
        self._layer = layer
        self._on_layer_update()

    @property
    def curating(self) -> bool:
        return self._curating

    @curating.setter
    def curating(self, curating: bool):
        if self._curating == curating:
            return

        if curating is True:
            self._on_start_curating()
        else:
            self._on_stop_curating()

        self._curating = curating

    def _expand_selected_labels_widget_function(
        self,
        pbar: widgets.ProgressBar,
        labels_to_expand: str,
        expansion_amount: int = 3,
    ) -> FunctionWorker[LayerDataTuple]:
        label_values_to_expand = labels_to_expand.replace(" ", "").split(",")
        label_values_to_expand = [int(value) for value in label_values_to_expand]

        # get the values from the selected labels layer
        label_image = self.layer.data
        label_layer_name = self.layer.name

        @thread_worker(connect={"returned": pbar.hide})
        def _expand_selected_labels() -> LayerDataTuple:

            new_labels = expand_selected_labels(
                label_image=label_image,
                label_values_to_expand=label_values_to_expand,
                expansion_amount=expansion_amount,
            )

            layer_kwargs = {"name": label_layer_name}
            return (new_labels, layer_kwargs, "labels")

        # show progress bar and return worker
        pbar.show()

        return _expand_selected_labels()

    def _on_layer_update(self):
        """callback function that is called when self.layer is updated"""
        pass

    def _toggle_curating(self, labels_layer: napari.layers.Labels):
        self.layer = labels_layer
        self.curating = True

    def _get_valid_labels_layers(self, combo_box) -> List[napari.layers.Labels]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _on_start_curating(self):
        self._label_expansion_widget.native.setVisible(True)

    def _on_stop_curating(self):
        self._label_expansion_widget.native.setVisible(False)
