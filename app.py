from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('../output.mp4', -1, 10.0, (640,480))
while cap.isOpened():
	ret, frame = cap.read()
	if ret == True:
		output = detector.detect_faces(frame)
		if len(output) == 2:
			if output:
				boxes = [output[i]['box'] for i in range(len(output))]
				cropped_images = []
				for box in boxes:
					#cv2.rectangle(frame, (box[0], box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0), 2)
					cropped_im = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
					cropped_images.append([cropped_im, box])
				im1, im2 = cropped_images[0][0],cropped_images[1][0]
				im1 = cv2.resize(im1, (cropped_images[1][1][2], cropped_images[1][1][3]))
				im2 = cv2.resize(im2, (cropped_images[0][1][2], cropped_images[0][1][3]))
				frame[cropped_images[0][1][1]:cropped_images[0][1][1]+cropped_images[0][1][3], cropped_images[0][1][0]:cropped_images[0][1][0]+cropped_images[0][1][2]] = im2
				frame[cropped_images[1][1][1]:cropped_images[1][1][1]+cropped_images[1][1][3], cropped_images[1][1][0]:cropped_images[1][1][0]+cropped_images[1][1][2]] = im1
		cv2.imshow('Image', frame)
		out.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
		
cap.release()
out.release()
cv2.destroyAllWindows()
