# !!!should be placed in GMR root!!!
# need to do afterwards: 30fps to 50 fps
# delete some bad data

import os
import cv2
import glob

VIDEO_ROOT = "usage/soccerkicks"
LABEL_FILE = "kick_labels.txt"

# 自动生成空label文件
video_dirs = sorted(glob.glob(os.path.join(VIDEO_ROOT, "*")))

names = []

for d in video_dirs:
    if not os.path.isdir(d):
        continue

    base = os.path.basename(d)

    mp4 = os.path.join(
        d,
        f"{base}_3_incam_global_horiz.mp4"
    )

    if os.path.exists(mp4):
        names.append(base)

with open(LABEL_FILE, "w") as f:
    for n in names:
        f.write(f"{n}:,\n")

print(f"Empty label file written to {LABEL_FILE}")

# 播放视频
for name in names:

    mp4 = os.path.join(
        VIDEO_ROOT,
        name,
        f"1_incam.mp4"
    )

    cap = cv2.VideoCapture(mp4)

    frame_id = 0

    print("=" * 60)
    print(f"Playing: {name}")
    print("SPACE: pause/resume")
    print("A/D: prev/next frame when paused")
    print("Q: next video")
    print("=" * 60)

    paused = False

    while True:

        if not paused:
            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

        show = frame.copy()

        cv2.putText(
            show,
            f"Frame: {frame_id}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        cv2.imshow("soccerkicks", show)

        key = cv2.waitKey(30) & 0xFF

        # q -> next video
        if key == ord('q'):
            break

        # pause
        elif key == ord(' '):
            paused = not paused

        # prev frame
        elif paused and key == ord('a'):
            frame_id = max(0, frame_id - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = cap.read()
            frame_id += 1

        # next frame
        elif paused and key == ord('d'):
            ret, frame = cap.read()
            if ret:
                frame_id += 1

    cap.release()

cv2.destroyAllWindows()