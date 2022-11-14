import os
import jsonfiler
import sys
from tools.toolbox_ui import Ui_Dialog
import cv2
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QImage, QPixmap

default_mot_path = "C:\Database\MOT\MOT20"
save_img = False


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.mot_path.setText(default_mot_path)

        self.stop_display = False
        self.work = False
        self.ui.label_2.clear()

    def get_set_com(self, set_name):
        com_dict = {"Train": "1",
                    "Test": "2",
                    "All": "3"}

        return com_dict.get(set_name, '1')

    def get_imglist(self, show=True):
        seq_path = self.ui.seq_box.currentText()
        show_content = [seq_path]

        if show:
            for img_list_path in show_content:
                print(img_list_path)
                img_list = os.listdir(img_list_path)
                for i, img_path in enumerate(img_list):
                    print(img_path, end=' ')
                    if i % 20 == 0 and i != 0:
                        print()
                print()

        return show_content

    def play_video(self, ground_truth=False, show_det=False):
        if self.work:
            self.stop_display = True
            self.work = False
            return
        self.work = True

        img_list_path = self.get_imglist(show=False)[0]
        if img_list_path is '':
            self.ui.label_2.setText("No sequence selected")
            print("No sequence selected")
            self.work = False
            return
        Msg = "Display video seq: " + img_list_path
        print(Msg)

        img_list = os.listdir(img_list_path)
        if ground_truth:
            ground_truth_dict = {}
            if "train" in img_list_path:
                gt_path = os.path.join(os.path.dirname(img_list_path), "gt\\gt.txt")
                with open(gt_path, 'r') as f:
                    gt_list = f.readlines()
                for line in gt_list:
                    i = int(line.split(',')[0])
                    if i not in ground_truth_dict.keys():
                        ground_truth_dict[i] = [line.split(',')[2:6]]
                    else:
                        ground_truth_dict[i].extend([line.split(',')[2:6]])
                jsonfiler.dump(ground_truth_dict, os.path.join(os.path.dirname(img_list_path), "gt", "gt.json"),
                               indent=4)
            else:
                self.ui.label_2.setText("Only training data has ground truth")
                print("Only training data has ground truth")
                self.work = False
                return

        if show_det:
            det_dict = {}
            gt_path = os.path.join(os.path.dirname(img_list_path), "det\\det.txt")
            with open(gt_path, 'r') as f:
                gt_list = f.readlines()
            for line in gt_list:
                i = int(line.split(',')[0])
                if i not in det_dict.keys():
                    det_dict[i] = [line.split(',')[2:6]]
                else:
                    det_dict[i].extend([line.split(',')[2:6]])

        for i, img_name in enumerate(img_list):
            img_path = os.path.join(img_list_path, img_name)
            img = cv2.imread(img_path)
            if ground_truth:
                for box in ground_truth_dict[i + 1]:
                    pt1 = int(box[0]), int(box[1])
                    pt2 = int(box[0]) + int(box[2]), int(box[1]) + int(box[3])
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255))
            if show_det:
                for box in det_dict[i + 1]:
                    pt1 = int(float(box[0])), int(float(box[1]))
                    pt2 = int(float(box[0])) + int(float(box[2])), int(float(box[1])) + int(float(box[3]))
                    cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

            show = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))
            cv2.waitKey(30)

            # Image save
            if save_img:
                img_name = os.path.basename(img_path)
                out_path = os.path.join('out', 'vis', img_name)
                cv2.imwrite(out_path, img)

            # Detect if the Stop key has been pressed
            if self.stop_display:
                self.stop_display = False
                break

        self.ui.label_2.clear()
        self.work = False

    def check_image_path(self, seq_path):
        if not os.path.exists(seq_path):
            err_info = "Cannot find image dir: "+seq_path+"\nPlease download full data set from MOT website."
            self.ui.label_2.setText(err_info)
            print(err_info)
            return False
        return True

    def fresh_seq(self):
        self.ui.seq_box.clear()
        if self.ui.mot_path.text() is '':
            mot_root_dir = default_mot_path
        else:
            mot_root_dir = self.ui.mot_path.text()

        mot_train_dir = os.path.join(mot_root_dir, "train")
        mot_test_dir = os.path.join(mot_root_dir, "test")

        err_info = ""
        if "train" not in os.listdir(mot_root_dir):
            err_info += "Cannot find training data set in: " + mot_train_dir + "\n"
            print("Cannot find training data set in:", mot_train_dir)

        if "test" not in os.listdir(mot_root_dir):
            err_info += "Cannot find testing data set in:" + mot_test_dir + "\n"
            print("Cannot find testing data set in:", mot_test_dir)

        if err_info != "":
            self.ui.label_2.setText(err_info)
            return

        ori_train_lists = [os.path.join(mot_train_dir, path, "img1") for path in os.listdir(mot_train_dir)]
        ori_test_lists = [os.path.join(mot_test_dir, path, "img1") for path in os.listdir(mot_test_dir)]

        set_name = self.ui.set_box.currentText()
        command = self.get_set_com(set_name)
        if command == "1":  # Train
            for i, seq in enumerate(ori_train_lists):
                if not self.check_image_path(seq):
                    continue
                self.ui.seq_box.addItem(seq)
        elif command == "2":  # Test
            for i, seq in enumerate(ori_test_lists):
                if not self.check_image_path(seq):
                    continue
                self.ui.seq_box.addItem(seq)
        elif command == "3":  # All
            for i, seq in enumerate(ori_train_lists):
                if not self.check_image_path(seq):
                    continue
                self.ui.seq_box.addItem(seq)
            for i, seq in enumerate(ori_test_lists):
                if not self.check_image_path(seq):
                    continue
                self.ui.seq_box.addItem(seq)

    def displayBT(self):
        self.play_video()

    def gtBT(self):
        self.play_video(ground_truth=True)

    def detBT(self):
        self.play_video(show_det=True)

    def stopBT(self):
        self.stop_display = True
        self.work = False


def qt5_init():
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())


def main():
    qt5_init()


if __name__ == '__main__':
    main()
