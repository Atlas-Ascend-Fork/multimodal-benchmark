# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Qt OpenGL label helpers for the CLOCs viewer."""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class GLLabelItem(GLGraphicsItem):
    def __init__(self, pos=None, text=None, color=None, font=None):
        super().__init__()
        self.color = QtCore.Qt.white if color is None else color
        self.text = text
        self.pos = pos
        self.font = QtGui.QFont() if font is None else font

    def set_gl_view_widget(self, gl_view_widget):
        self.gl_view_widget = gl_view_widget

    def set_data(self, pos, text, color, font=None):
        self.text = text
        self.pos = pos
        self.color = color
        if font is not None:
            self.font = font
        self.update()

    def paint(self):
        self.gl_view_widget.qglColor(self.color)
        if self.pos is None or self.text is None:
            return
        if isinstance(self.pos, (list, tuple, np.ndarray)):
            for position, text in zip(self.pos, self.text):
                self.gl_view_widget.renderText(*position, text, self.font)
        else:
            self.gl_view_widget.renderText(*self.pos, self.text, self.font)


def pltcolor_to_qtcolor(color):
    color_map = {
        "r": QtCore.Qt.red,
        "g": QtCore.Qt.green,
        "b": QtCore.Qt.blue,
        "k": QtCore.Qt.black,
        "w": QtCore.Qt.white,
        "y": QtCore.Qt.yellow,
        "c": QtCore.Qt.cyan,
        "m": QtCore.Qt.magenta,
    }
    return color_map.get(color)
