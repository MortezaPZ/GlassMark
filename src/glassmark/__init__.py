"""GlassMark phase 1: glare suppression and marker isolation on glass photos.

The package is CPU only (OpenCV + NumPy) and contains:

* :mod:`glassmark.synth` - synthetic oily-glass scenes and their ground truth;
* :mod:`glassmark.glare` - stages A and B (specular regions, glare versus ink);
* :mod:`glassmark.suppress` - stage C (rebuild glared ink, inpaint the rest);
* :mod:`glassmark.markers` - stage D (keep only marker-like blobs);
* :mod:`glassmark.pipeline` - runs the stages and writes the artefacts;
* :mod:`glassmark.checks` - ground-truth comparison used by the test suite.
"""

__all__ = [
    "cases",
    "checks",
    "config",
    "geom",
    "glare",
    "imgio",
    "markers",
    "pipeline",
    "report",
    "suppress",
    "synth",
]

__version__ = "0.1.0"
