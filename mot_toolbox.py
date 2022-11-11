import os
import cv2
import jsonfiler

mot_root_dir = "G:/Database/MOT/MOT20"
voc_output_dir = "voc"
save_img = False

mot_train_dir = os.path.join(mot_root_dir, "train")
mot_test_dir = os.path.join(mot_root_dir, "test")
ori_train_lists = [os.path.join(mot_train_dir, path, "img1") for path in os.listdir(mot_train_dir)]
ori_test_lists = [os.path.join(mot_test_dir, path, "img1") for path in os.listdir(mot_test_dir)]


def check_seq():
    print("Train:", ori_train_lists)
    print("Test:", ori_test_lists)


def get_imglist(show=True):
    show_content = []
    print("[1]: Train [2]: Test [3]: All")
    command = input(">>")
    if command == "1":  # Train
        # Display caption
        for i, seq in enumerate(ori_train_lists):
            print("[%d] %s" % (i, seq), end=' ')
        print("[%d] All" % len(ori_train_lists))
        # Select a sequence
        command_num = int(input(">>"))
        if command_num < len(ori_train_lists):
            show_content.append(ori_train_lists[command_num])
        elif command_num == len(ori_train_lists):
            show_content.extend(ori_train_lists)
    elif command == "2":  # Test
        # Display caption
        for i, seq in enumerate(ori_test_lists):
            print("[%d] %s" % (i, seq), end=' ')
        print("[%d] All" % len(ori_test_lists))
        # Select a sequence
        command_num = int(input(">>"))
        if command_num < len(ori_test_lists):
            show_content.append(ori_test_lists[command_num])
        elif command_num == len(ori_test_lists):  # All
            show_content.extend(ori_test_lists)
    elif command == "3":  # All
        show_content.extend(ori_train_lists)
        show_content.extend(ori_test_lists)

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


def help_show():
    print("Select a function:")
    print("[1]: check_seq")
    print("[2]: show image list")
    print("[3]: display video sequence")
    print("[4]: display ground truth")
    print("[5]: display detection")


def show_video_sequence(scaling_factor=0.5, ground_truth=False, show_det=False):
    img_list_path = get_imglist(show=False)[0]
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
            jsonfiler.dump(ground_truth_dict, os.path.join(os.path.dirname(img_list_path), "gt", "gt.json"), indent=4)
        else:
            input("Only training data has ground truth, press any key to return")
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
            for box in ground_truth_dict[i+1]:
                pt1 = int(box[0]), int(box[1])
                pt2 = int(box[0]) + int(box[2]), int(box[1]) + int(box[3])
                cv2.rectangle(img, pt1, pt2, (0, 0, 255))
        if show_det:
            for box in det_dict[i+1]:
                pt1 = int(float(box[0])), int(float(box[1]))
                pt2 = int(float(box[0])) + int(float(box[2])), int(float(box[1])) + int(float(box[3]))
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

        # Resize the frame
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor,
                           interpolation=cv2.INTER_AREA)

        cv2.imshow("MOT frame", img)

        # Image save
        if save_img:
            img_name = os.path.basename(img_path)
            out_path = os.path.join('out', 'vis', img_name)
            cv2.imwrite(out_path, img)
        cv2.waitKey(30)
    # Close all active windows
    cv2.destroyAllWindows()


def main():
    while True:
        help_show()
        command = input(">>")
        if command == "q":
            print("Bye")
            break
        elif command == "1":
            check_seq()
        elif command == "2":
            get_imglist()
        elif command == "3":
            show_video_sequence()
        elif command == "4":
            show_video_sequence(ground_truth=True)
        elif command == "5":
            show_video_sequence(show_det=True)


if __name__ == '__main__':
    main()

