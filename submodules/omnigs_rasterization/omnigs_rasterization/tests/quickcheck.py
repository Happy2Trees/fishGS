from __future__ import annotations


def main() -> None:
    try:
        from omnigs_rasterization import _C  # type: ignore
    except Exception as e:  # pragma: no cover
        print("Extension import failed:", e)
        raise
    else:
        print("Extension module loaded:", _C.__name__)


if __name__ == "__main__":
    main()

