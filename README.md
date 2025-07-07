
avatar-session-jambase
======================






GUI for the demo



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

## Common Issues
### This application failed to start because no Qt platform plugin could be initialized
"""
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: wayland-egl, vkkhrdisplay, xcb, minimal, linuxfb, minimalegl, wayland, eglfs, vnc, offscreen.
"""
To resolve this issue, you need to install the `xcb-cursor` package. You can do this using your package manager. For example, on Ubuntu, you can run:

```bash
sudo apt-get install libxcb-cursor-dev
```