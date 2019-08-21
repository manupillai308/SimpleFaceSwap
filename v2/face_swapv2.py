import cv2
import dlib
import numpy as np


def draw_mask(img, points):
	for i in range(16):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	for i in range(48, 67):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	cv2.line(img, tuple(points[60]), tuple(points[67]), (255,255,255), 1)
	for i in range(31, 35):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	for i in range(27, 30):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	for i in range(17, 21):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	for i in range(22, 26):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	for i in range(36, 41):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	cv2.line(img, tuple(points[36]), tuple(points[41]), (255,255,255), 1)
	for i in range(42, 47):
		cv2.line(img, tuple(points[i+1]), tuple(points[i]), (255,255,255), 1)
	cv2.line(img, tuple(points[47]), tuple(points[42]), (255,255,255), 1)

def applyAffineTransform(src, srcTri, dstTri, size) :
	
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
	
	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

	return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
	if point[0] < rect[0] :
		return False
	elif point[1] < rect[1] :
		return False
	elif point[0] > rect[0] + rect[2] :
		return False
	elif point[1] > rect[1] + rect[3] :
		return False
	return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
	#create subdiv
	subdiv = cv2.Subdiv2D(rect);
	# Insert points into subdiv
	for p in points:
		subdiv.insert(tuple(p))
	triangleList = subdiv.getTriangleList();
	delaunayTri = []
	
	pt = []    
		
	for t in triangleList:        
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))
		
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])        
		
		if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
			ind = []
			#Get face-points (from 68 face detector) by coordinates
			for j in range(0, 3):
				for k in range(0, len(points)):                    
					if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
						ind.append(k)    
			# Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
			if len(ind) == 3:                                                
				delaunayTri.append((ind[0], ind[1], ind[2]))
		
		pt = []        
			
	
	return delaunayTri
		

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))

	# Offset points by left top corner of the respective rectangles
	t1Rect = [] 
	t2Rect = []
	t2RectInt = []

	for i in range(0, 3):
		t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
		t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
		t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


	# Get mask by filling triangle
	mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	#img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
	
	size = (r2[2], r2[3])

	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
	
	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
	 
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
	
def swap(img, faces, img1Warped):
	img1 = img
	img2 = img
	
	points1 = faces[0]
	points2 = faces[1]
	
	hull1 = []
	hull2 = []

	hullIndex = cv2.convexHull(points2, returnPoints = False)
		  
	for i in range(0, len(hullIndex)):
		hull1.append(points1[int(hullIndex[i])])
		hull2.append(points2[int(hullIndex[i])])
	
	sizeImg2 = img2.shape    
	rect = (0, 0, sizeImg2[1], sizeImg2[0])
	hull2 = list((np.array(hull2)[np.array(hull2) > [0,0]]).reshape((-1,2)))
	hull1 = list((np.array(hull1)[np.array(hull1) > [0,0]]).reshape((-1,2)))
	dt = calculateDelaunayTriangles(rect, hull2)
	if len(dt) == 0:
		return None
	
	for i in range(0, len(dt)):
		t1 = []
		t2 = []
		
		for j in range(0, 3):
			t1.append(hull1[dt[i][j]])
			t2.append(hull2[dt[i][j]])
		warpTriangle(img1, img1Warped, t1, t2)
	
	hull8U = []
	for i in range(0, len(hull2)):
		hull8U.append((hull2[i][0], hull2[i][1]))
	
	mask = np.zeros(img2.shape, dtype = img2.dtype)  
	
	cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
	
	r = cv2.boundingRect(np.float32([hull2]))    
	
	center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
	
	return cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)


predictor_path = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cam = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
writer = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))

while(cam.isOpened()):
	ret, img = cam.read()
	if ret:
		dets = detector(img[...,[2,1,0]], 1)
		faces = []
		for k, d in enumerate(dets):
			shape = predictor(img[...,[2,1,0]], d)
			#cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0,255,0), 2)
			points = np.empty((68, 2), dtype=int)
			for i in range(68):
				v = [shape.part(i).x, shape.part(i).y]
				#cv2.circle(img, (v[0], v[1]), 2, (255,255,255), -1)
				points[i][0], points[i][1] = v[0], v[1]
			faces.append(points)
			#draw_mask(img, points)
		if len(faces) >= 2:
			face1 = swap(img, faces, np.copy(img))
			face2 = swap(img, [faces[1], faces[0]], np.copy(face1))
			if face1 is None or face2 is None:
				continue
			img = face1 + face2 - img
		cv2.imshow("Camera Window", img)
		writer.write(img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break


cam.release()
writer.release()
cv2.destroyAllWindows()
