import cv2
import os
import glob

image_folder = r'C:\Users\159fe\code\image-processing\divided-images'
video_name = 'video.avi'
current_image = 0
start = 0
end = 100

images = sorted(glob.glob(f"{image_folder}\\*"), key = lambda x: int(x.split('\\')[-1].split('.')[0].split('e')[1]))[start:end][current_image:]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1/3, (width,height))
count = 1

for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    cv2.putText(frame, image.split('\\')[6][:-4] + " t = " + str(count*30) + " ms", (50, height-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    video.write(frame)
    count += 1

cv2.destroyAllWindows()
video.release()