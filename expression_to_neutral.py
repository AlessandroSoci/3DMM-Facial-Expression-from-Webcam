from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
import sys
import qdarkstyle

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainW = MainWindow(True)
    mainW.activate()
    sys.exit(app.exec_())
