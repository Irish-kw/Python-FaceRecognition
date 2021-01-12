import cv2

def detectFace(img_name, classifier, scaleFactor, minNeighbors):
    color = (0, 255, 0)  # 框的顏色
    filename = img_name.split(".")[0]  # output filename
    img = cv2.imread(img_name)  # 讀取圖檔
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉為灰階影像

    # 偵測人臉函數
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    if classifier == 1:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
    if classifier == 2:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")
    if classifier == 3:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    if classifier == 4:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    if classifier == 5:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml")
    if classifier == 6:
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # image	待檢測圖片，一般為灰階影像，以便加快偵測速度
    # scaleFactor	在前後兩次相繼的掃描中，搜索範圍的比例係數，默認值為 1.1
    # minNeighbors	構成偵測目標的相鄰矩形的最小個數，默認值為 3
    # minSize & maxSize	用來限制得到的目標區域範圍

    faceRects = face_classifier.detectMultiScale(grayImg, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # 大於 0 則檢測到人臉
    if len(faceRects):
        # 框出每一張人臉
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)

    # cv2.imshow('Original image', img)
    #
    # # 按下Q才離開
    # k = cv2.waitKey(0) & 0xFF
    # if k == ord('q') or k == ord('Q'):
    #     print('exit')
    #     cv2.destroyAllWindows()

    cv2.imwrite(filename + "_Classiifier" + str(classifier) + "_face.jpg", img)
    return None

detectFace('human2.jpg', classifier=3, scaleFactor=1.1, minNeighbors=2)
