from __future__ import annotations

import os
from pathlib import Path
from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required to build the OmniGS extension. Install torch matching your CUDA runtime."
    ) from e


PKG_ROOT = Path(__file__).resolve().parent


def _extra_compile_args() -> dict[str, list[str]]:
    cxx_args = ["-O3", "-std=c++17"]
    nvcc_args = ["-O3", "--expt-relaxed-constexpr", "-lineinfo", "-std=c++17"]
    return {"cxx": cxx_args, "nvcc": nvcc_args}


sources = [
    PKG_ROOT / "omnigs_rasterization" / "ext.cpp",
    PKG_ROOT / "src" / "rasterize_points.cu",
    PKG_ROOT / "cuda_rasterizer" / "rasterizer_impl.cu",
    PKG_ROOT / "cuda_rasterizer" / "forward.cu",
    PKG_ROOT / "cuda_rasterizer" / "backward.cu",
]

include_dirs = [
    str(PKG_ROOT),  # so includes like "include/..." and "cuda_rasterizer/..." resolve
    str(PKG_ROOT / "third_party"),  # contains glm/
]

ext_modules = [
    CUDAExtension(
        name="omnigs_rasterization._C",
        sources=[str(s) for s in sources],
        include_dirs=include_dirs,
        extra_compile_args=_extra_compile_args(),
    )
]

setup(
    name="omnigs-rasterization",
    version="0.0.1",
    description="PyTorch bindings for the OmniGS rasterizer (self-contained)",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

