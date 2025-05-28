from PyQt5.QtWidgets import *
app = QApplication([])

g_label = QLabel('Primary station tx antenna gain (dB)')
g_label.show()
Gt = QLineEdit()
Gt.show()

app.exec()