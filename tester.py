import cv2
from skd import SKD


def image_detection(img_path):
    skd = SKD()
    img = cv2.imread(img_path)
    img = skd.detect_img(img)
    cv2.imshow("Img", img)
    cv2.waitKey(0)


def video_detection():
    skd = SKD()
    # cap = cv2.VideoCapture("../minhanh_extra_gestures/raw_video/S3.mp4")
    cap = cv2.VideoCapture(0)
    count = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    # cap.release()
    # return

    # old_time = time.time()
    while True:
        timer = cv2.getTickCount()
        success, frame = cap.read()

        if not success:
            continue

        # if count < 200:
        #     count += 1
        #     continue
        frame = cv2.flip(frame, 1)
        frame = skd.detect_video(frame, multi_hand=False)

        # new_time = time.time()
        # fps = int(1 / (new_time - old_time))
        fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
        # old_time = new_time
        cv2.putText(frame, f"FPS: {fps}", (5, 30), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Img", frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(detectVideo):
    if detectVideo:
        video_detection()
    else:
        image_detection('sample_img/test_img2.jpg')


if __name__ == '__main__':
    main(True)
