# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Register_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training
from PyQt5 import QtCore, QtGui, QtWidgets
from preprocess import preprocesses
import Database as db
import cv2
import os
import time



class Ui_RegisterWindow(object):
    def __init__(self):
        self.counter = 0

    def setupUi(self, RegisterWindow):
        RegisterWindow.setObjectName("RegisterWindow")
        RegisterWindow.resize(500, 400)
        self.centralwidget = QtWidgets.QWidget(RegisterWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.PreProcess_button = QtWidgets.QPushButton(self.centralwidget)
        self.PreProcess_button.setGeometry(QtCore.QRect(180, 220, 150, 51))
        self.PreProcess_button.setStyleSheet("font: 63 12pt \"Bahnschrift\";")
        self.PreProcess_button.setObjectName("PreProcess_button")
        self.PreProcess_button.clicked.connect(self.train_image)
        self.Generate_Data = QtWidgets.QPushButton(self.centralwidget)
        self.Generate_Data.setGeometry(QtCore.QRect(180, 160, 150, 51))
        self.Generate_Data.setStyleSheet("font: 63 12pt \"Bahnschrift\";")
        self.Generate_Data.setObjectName("Generate_Data")
        self.Generate_Data.clicked.connect(self.generatedata_method)
        self.Nameinput = QtWidgets.QLineEdit(self.centralwidget)
        self.Nameinput.setGeometry(QtCore.QRect(170, 40, 250, 40))
        self.Nameinput.setObjectName("Nameinput")
        self.MatricInput = QtWidgets.QLineEdit(self.centralwidget)
        self.MatricInput.setGeometry(QtCore.QRect(170, 80, 250, 40))
        self.MatricInput.setObjectName("MatricInput")
        self.NameText = QtWidgets.QLabel(self.centralwidget)
        self.NameText.setGeometry(QtCore.QRect(120, 40, 50, 40))
        self.NameText.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.NameText.setText("")
        self.NameText.setPixmap(QtGui.QPixmap("../Integration stage 1/name.png"))
        self.NameText.setScaledContents(True)
        self.NameText.setObjectName("NameText")
        self.Matric_image = QtWidgets.QLabel(self.centralwidget)
        self.Matric_image.setGeometry(QtCore.QRect(120, 80, 50, 40))
        self.Matric_image.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.Matric_image.setText("")
        self.Matric_image.setPixmap(QtGui.QPixmap("key-icon.png"))
        self.Matric_image.setScaledContents(True)
        self.Matric_image.setObjectName("Matric_image")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 220, 150, 51))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        RegisterWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RegisterWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 387, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuInfo = QtWidgets.QMenu(self.menubar)
        self.menuInfo.setObjectName("menuInfo")
        RegisterWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RegisterWindow)
        self.statusbar.setObjectName("statusbar")
        RegisterWindow.setStatusBar(self.statusbar)
        self.actionhelp = QtWidgets.QAction(RegisterWindow)
        self.actionhelp.setObjectName("actionhelp")
        self.actionNew = QtWidgets.QAction(RegisterWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionClose = QtWidgets.QAction(RegisterWindow)
        self.actionClose.setObjectName("actionClose")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionClose)
        self.menuInfo.addAction(self.actionhelp)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())

        self.retranslateUi(RegisterWindow)
        QtCore.QMetaObject.connectSlotsByName(RegisterWindow)

    def retranslateUi(self, RegisterWindow):
        _translate = QtCore.QCoreApplication.translate
        RegisterWindow.setWindowTitle(_translate("RegisterWindow", "MainWindow"))
        self.PreProcess_button.setText(_translate("RegisterWindow", "SUBMIT"))
        self.Generate_Data.setText(_translate("RegisterWindow", "REGISTER FACE"))
        self.Nameinput.setPlaceholderText(_translate("RegisterWindow", "ENTER YOUR NAME"))
        self.MatricInput.setPlaceholderText(_translate("RegisterWindow", "ENTER MATRICULATION NO."))
        self.menuFile.setTitle(_translate("RegisterWindow", "File"))
        self.menuInfo.setTitle(_translate("RegisterWindow", "Info"))
        self.actionhelp.setText(_translate("RegisterWindow", "help"))
        self.actionNew.setText(_translate("RegisterWindow", "New"))
        self.actionClose.setText(_translate("RegisterWindow", "Close"))



    def generatedata_method(self):
        total_pic = 0
        print('Capturing Name Entered')
        #name = input()
        #taking name input from the GUI
        name= self.Nameinput.text()
        matric_number= self.MatricInput.text()
        db.regStud(name,matric_number)

        os.chdir('train_img')
        os.makedirs(name)

        # change the cascader file path to the absolute path on your system
        fa = cv2.CascadeClassifier(
            'C:/Users/HP/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        # Start image capture on webcam. Port 0
        cap = cv2.VideoCapture(0)
        # sleep for 2 seconds to warm up camera
        time.sleep(2)
        cv2.namedWindow("frame")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fa.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # cropped=frame[y:2*(y+h),x:2*(x+w)]
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,cv2.LINE_AA)
                total_pic += 1
                cv2.imwrite(name + '/' + 'ActiOn_' + str(total_pic) + '.jpg', frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(100)

            if total_pic > 50:
                break

        cap.release()
        cv2.destroyAllWindows()

    def train_image(self):


        print("Tried to train image")
        input_datadir = './train_img'
        output_datadir = './pre_img'

        obj = preprocesses(input_datadir, output_datadir)
        num_images_total, num_successfully_aligned = obj.collect_data()

        print('Total number of images: %d' % num_images_total)
        print('Number of successfully aligned images: %d' % num_successfully_aligned)

        self.train_image_2()


    def train_image_2(self):


        self.counter += 10
        self.progressBar.setValue(self.counter)
        print("tried even harder")
        datadir = './pre_img'
        modeldir = './model/20170511-185253.pb'
        classifier_filename = './class/classifier.pkl'
        print("Model Training Started")

        self.counter += 20
        self.progressBar.setValue(self.counter)
        obj = training(datadir, modeldir, classifier_filename)
        get_file = obj.main_train()

        self.counter += 50
        self.progressBar.setValue(self.counter)
        print('Saved classifier model to file "%s"' % get_file)
        self.counter += 20
        self.progressBar.setValue(self.counter)
        sys.exit("Model Training Complete and Mode saved")


        RegisterWindow.close()




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RegisterWindow = QtWidgets.QMainWindow()
    ui = Ui_RegisterWindow()
    ui.setupUi(RegisterWindow)
    RegisterWindow.show()
    sys.exit(app.exec_())
