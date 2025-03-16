mainWindow_styleSheet = """
            QMainWindow {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #6C6C6C, stop: 1 #5B5B5B);
            }
        """

combo_styleSheet = """
            QComboBox {
                border: 1px solid gray;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 5em;
            }

            QComboBox:editable {
                background: white;
            }

            QComboBox:!editable, QComboBox::drop-down:editable {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                            stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
            }

            /* QComboBox gets the "on" state when the popup is open */
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,
                                            stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);
            }

            QComboBox:on { /* shift the text when the popup opens */
                padding-top: 3px;
                padding-left: 4px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;

                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid; /* just a single line */
                border-top-right-radius: 3px; /* same radius as the QComboBox */
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                border: 2px solid darkgray;
                selection-background-color: lightgray;
            }
        """
slider_styleSheet = """
                QSlider::groove:horizontal {
                    border: 1px solid #999999;
                    height: 8px; /* the groove expands to the size of the slider by default. by giving it a height, it has a fixed size */
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                    margin: 2px 0;
                }

                QSlider::handle:horizontal {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                    border: 1px solid #5c5c5c;
                    width: 18px;
                    margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */
                    border-radius: 3px;
                }
        """
        
button_styleSheet = """QPushButton{
            background-color: gray;
            border-style: outset;
            border-radius: 10px;
            border-color: beige;
            padding: 6px;
        }
        QPushButton:pressed {
            background-color: lightgray;
            border-style: inset;
        }"""