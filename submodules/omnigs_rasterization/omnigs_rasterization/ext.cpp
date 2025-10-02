#include <torch/extension.h>

#include "include/rasterize_points.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch bindings for the OmniGS Gaussian rasterizer (self-contained)";

    m.def(
        "rasterize_gaussians",
        &RasterizeGaussiansCUDA,
        py::arg("background"),
        py::arg("means3d"),
        py::arg("colors"),
        py::arg("opacity"),
        py::arg("scales"),
        py::arg("rotations"),
        py::arg("scale_modifier"),
        py::arg("cov3d_precomp"),
        py::arg("viewmatrix"),
        py::arg("projmatrix"),
        py::arg("tan_fovx"),
        py::arg("tan_fovy"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("sh"),
        py::arg("degree"),
        py::arg("campos"),
        py::arg("prefiltered"),
        py::arg("camera_type") = 1,
        py::arg("render_depth") = false,
        "Forward rasterization pass"
    );

    m.def(
        "rasterize_gaussians_backward",
        &RasterizeGaussiansBackwardCUDA,
        py::arg("background"),
        py::arg("means3d"),
        py::arg("radii"),
        py::arg("colors"),
        py::arg("scales"),
        py::arg("rotations"),
        py::arg("scale_modifier"),
        py::arg("cov3d_precomp"),
        py::arg("viewmatrix"),
        py::arg("projmatrix"),
        py::arg("tan_fovx"),
        py::arg("tan_fovy"),
        py::arg("dL_dout_color"),
        py::arg("sh"),
        py::arg("degree"),
        py::arg("campos"),
        py::arg("geom_buffer"),
        py::arg("r"),
        py::arg("binning_buffer"),
        py::arg("image_buffer"),
        py::arg("camera_type") = 1,
        "Backward rasterization pass"
    );

    m.def(
        "mark_visible",
        &markVisible,
        py::arg("means3d"),
        py::arg("viewmatrix"),
        py::arg("projmatrix"),
        py::arg("camera_type") = 1,
        "Visibility query"
    );
}

