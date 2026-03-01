from __future__ import annotations


def test_package_main_importable() -> None:
    from shejianplus.main import main

    assert callable(main)