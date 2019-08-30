#importing required modules
import os
import cv2,pickle
import face_recognition as fr
import numpy as np

#Function to create known face encodings
def create_known_face_encodings():
	known_face_encodings_list=[]
	known_names=[]
	font=cv2.FONT_HERSHEY_SIMPLEX
	#creating image's directory
	try:
		#getting current working directory
    		cwd=os.getcwd()
		#creating a directory to store dataset images
    		os.mkdir(cwd+"/dataset_images")
	except:
    		print()

	dataset_dir=os.getcwd()+"/dataset_images/"
	#print(dataset_dir)
	image_path=os.listdir(dataset_dir)
	for i in image_path:
		#accessing each image
    		image=dataset_dir+i
    		print(image)
    		known_face= fr.load_image_file(image)
    		#getting encodings of faces
    		known_face_encoding=fr.face_encodings(known_face)
    		if len(known_face_encoding) > 0:
        		known_face_encoding=fr.face_encodings(known_face)[0]
	    	else:
        		print("fail")
		#getting image names
    		image_name=i.split("_")[0]
    		known_face_encodings_list.append(known_face_encoding)
    		known_names.append(image_name)
		#print(known_face_encodings_list)
		#dumping encodings in encodings.txt in binary mode
	with open("encodings.txt",'wb') as file_data:
    		pickle.dump(known_face_encodings_list,file_data)
		#dumping image names in name.txt in binary mode
	with open("name.txt",'wb') as file_data:
    		pickle.dump(known_names,file_data)

#calling this function
#create_known_face_encodings()

#loading those binary files
font=cv2.FONT_HERSHEY_SIMPLEX
with open("encodings.txt",'rb') as file_data:
    known_face_encodings=pickle.load(file_data)
with open("name.txt",'rb') as file_data:
    known_names=pickle.load(file_data)
#print(known_face_encodings)
#taking image path as input from user
img_path=input("Enter the image path you want to match:-")

#reading image
img=cv2.imread(img_path)
process_this_img = True
#converting BGR image to RGB image
rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
if process_this_img:
    #gettting face locations
    face_locations=fr.face_locations(rgb_img)
    #getting face encodings
    current_face_encoding=fr.face_encodings(rgb_img,face_locations)
    for face_encoding in current_face_encoding:
	#compariong face with known faces
        name="unknown"
        matches=fr.compare_faces(known_face_encodings,face_encoding)
        #print(matches)
	#get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            accurate=(1.0-min(face_distances))*100
        else:
            accurate=max(face_distances)*100
process_this_frame= not process_this_img
for (top, right, bottom, left) in face_locations:
    #creating a rectangle around face in frame				
    cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
    #putting image name on the top of rectangle 
    cv2.putText(img, name, (left , top), font, 1.0, (255, 225, 0), 4)
    cv2.putText(img,str(accurate), (10,50), font, 1.0, (255, 225, 0), 4)		
#showing our image 
cv2.namedWindow('Live',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live', 600,600)
cv2.imshow("Live",img)
cv2.waitKey(0)		
cv2.destroyAllWindows()

