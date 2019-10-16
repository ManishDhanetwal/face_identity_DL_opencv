# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FirstLoginWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PyQt5 import QtCore, QtGui, QtWidgets
from Register_window import Ui_RegisterWindow
import tensorflow as tf
import Database as db
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle


class Ui_ptFirstLoginWindow(object):
    def register_window(self):
        self.Register_window= QtWidgets.QMainWindow()
        self.ui= Ui_RegisterWindow()
        self.ui.setupUi(self.Register_window)
        self.Register_window.show()

    def setupUi(self, ptFirstLoginWindow):
        ptFirstLoginWindow.setObjectName("ptFirstLoginWindow")
        ptFirstLoginWindow.resize(600, 500)
        ptFirstLoginWindow.setAutoFillBackground(False)
        ptFirstLoginWindow.setStyleSheet("background -color : rgb(233, 241, 255)")
        self.centralwidget = QtWidgets.QWidget(ptFirstLoginWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.classname = QtWidgets.QLineEdit(self.centralwidget)
        self.classname.setGeometry(QtCore.QRect(110, 220, 191, 31))
        self.classname.setStyleSheet("font: 75 12pt \"Arial\";")
        self.classname.setObjectName("classname")
        self.loadbutton = QtWidgets.QPushButton(self.centralwidget)
        self.loadbutton.setGeometry(QtCore.QRect(390, 220, 91, 31))
        self.loadbutton.setStyleSheet("font: 63 12pt \"Bahnschrift\";")
        self.loadbutton.setObjectName("loadbutton")
        self.loadbutton.clicked.connect(self.loadclass)
        self.Loginimage = QtWidgets.QLabel(self.centralwidget)
        self.Loginimage.setGeometry(QtCore.QRect(20, 9, 550, 200))
        self.Loginimage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Loginimage.setText("")
        self.Loginimage.setPixmap(QtGui.QPixmap("header.jpeg"))
        self.Loginimage.setScaledContents(True)
        self.Loginimage.setWordWrap(False)
        self.Loginimage.setObjectName("Loginimage")
        self.mark_attendance_but = QtWidgets.QPushButton(self.centralwidget)
        self.mark_attendance_but.setGeometry(QtCore.QRect(250, 290, 191, 61))
        self.mark_attendance_but.setStyleSheet("\n"
"font: 63 12pt \"Bahnschrift\";")
        self.mark_attendance_but.setObjectName("mark_attendance_but")
        self.mark_attendance_but.clicked.connect(self.mark_attendance_method)
        self.newuser_but = QtWidgets.QPushButton(self.centralwidget)
        self.newuser_but.setGeometry(QtCore.QRect(250, 350, 191, 61))
        self.newuser_but.setStyleSheet("font: 63 12pt \"Bahnschrift\";")
        self.newuser_but.setObjectName("newuser_but")
        self.newuser_but.clicked.connect(self.register_window)
        self.attedance_img = QtWidgets.QLabel(self.centralwidget)
        self.attedance_img.setGeometry(QtCore.QRect(180, 290, 71, 61))
        self.attedance_img.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.attedance_img.setText("")
        self.attedance_img.setPixmap(QtGui.QPixmap("attedance.jpg"))
        self.attedance_img.setScaledContents(True)
        self.attedance_img.setObjectName("attedance_img")
        self.newuser_img = QtWidgets.QLabel(self.centralwidget)
        self.newuser_img.setGeometry(QtCore.QRect(180, 350, 71, 61))
        self.newuser_img.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.newuser_img.setText("")
        self.newuser_img.setPixmap(QtGui.QPixmap("create_Account.jpg"))
        self.newuser_img.setScaledContents(True)
        self.newuser_img.setObjectName("newuser_img")
        ptFirstLoginWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ptFirstLoginWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        ptFirstLoginWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ptFirstLoginWindow)
        self.statusbar.setObjectName("statusbar")
        ptFirstLoginWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(ptFirstLoginWindow)
        self.actionClose.setStatusTip("")
        self.actionClose.setObjectName("actionClose")
        self.menuFile.addAction(self.actionClose)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(ptFirstLoginWindow)
        QtCore.QMetaObject.connectSlotsByName(ptFirstLoginWindow)

    def loadclass(self):
        classname= self.classname.text()
        db.newClass(classname)
       # self.classname.textChanged("CLass loaded")

    def retranslateUi(self, ptFirstLoginWindow):
        _translate = QtCore.QCoreApplication.translate
        ptFirstLoginWindow.setWindowTitle(_translate("ptFirstLoginWindow", "MainWindow"))
        self.classname.setStatusTip(_translate("ptFirstLoginWindow", "Enter the class name"))
        self.classname.setPlaceholderText(_translate("ptFirstLoginWindow", "Enter the Class"))
        self.loadbutton.setStatusTip(_translate("ptFirstLoginWindow", "Load into a specific class"))
        self.loadbutton.setText(_translate("ptFirstLoginWindow", "LOAD"))
        self.mark_attendance_but.setStatusTip(_translate("ptFirstLoginWindow", "Start Marking attendance"))
        self.mark_attendance_but.setText(_translate("ptFirstLoginWindow", "MARK ATTENDANCE"))
        self.newuser_but.setStatusTip(_translate("ptFirstLoginWindow", "Register a new user"))
        self.newuser_but.setText(_translate("ptFirstLoginWindow", "NEW USER"))
        self.menuFile.setTitle(_translate("ptFirstLoginWindow", "File"))
        self.actionClose.setText(_translate("ptFirstLoginWindow", "Close"))

    def mark_attendance_method(self):
        modeldir = './model/20170511-185253.pb'
        classifier_filename = './class/classifier.pkl'
        npy = './npy'
        train_img = "./train_img"

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                batch_size = 1000
                image_size = 182
                input_image_size = 160

                HumanNames = os.listdir(train_img)
                HumanNames.sort()

                print('Loading Modal')
                facenet.load_model(modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                video_capture = cv2.VideoCapture(0)
                c = 0

                print('Start Recognition')
                prevTime = 0
                while True:
                    ret, frame = video_capture.read()

                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

                    curTime = time.time() + 1  # calc fps
                    timeF = frame_interval

                    if (c % timeF == 0):
                        find_results = []

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]

                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                            for i in range(nrof_faces):
                                emb_array = np.zeros((1, embedding_size))

                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(
                                        frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                print(predictions)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                # print("predictions")
                                print(best_class_indices, ' with accuracy ', best_class_probabilities)

                                # print(best_class_probabilities)
                                if best_class_probabilities > 0.65:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                                  2)  # boxing face

                                    # plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    print('Result Indices: ', best_class_indices[0])
                                    print(HumanNames)
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = HumanNames[best_class_indices[0]]
                                            db.takeAttendance(result_names)
                                            db.showDB()
                                            cv2.putText(frame, result_names, (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            print('Alignment Failure')
                    # c+=1
                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ptFirstLoginWindow = QtWidgets.QMainWindow()
    ui = Ui_ptFirstLoginWindow()
    ui.setupUi(ptFirstLoginWindow)
    ptFirstLoginWindow.show()
    sys.exit(app.exec_())
