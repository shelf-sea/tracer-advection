# -*- coding: utf-8 -*-

from ess_tracer_transit import version


def test_version():
    """Double check version."""
    assert version.pkg_version == "20.1"
