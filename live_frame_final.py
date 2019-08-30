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
create_known_face_encodings()


font=cv2.FONT_HERSHEY_SIMPLEX
#loading those binary files
with open("encodings.txt",'rb') as file_data:
    known_face_encodings=pickle.load(file_data)
with open("name.txt",'rb') as file_data:
    known_names=pickle.load(file_data)
#print(known_face_encodings)
#creating instance to use camera 0 is passed to use primary webcam
cam=cv2.VideoCapture(0)
#length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
process_this_frame = True
while True:
    #reading frame
    frame=cam.read()[1]
    #converting BGR frame to RGB frame
    rgb_frame=frame[:,:,::-1]
    if process_this_frame:
    #gettting face locations in current frame
    	face_locations=fr.face_locations(rgb_frame)
    #getting face encodings in current frame
    	current_face_encoding=fr.face_encodings(rgb_frame,face_locations)
    	for face_encoding in current_face_encoding:
	#compariong face with known faces
            name="unknown"
            matches=fr.compare_faces(known_face_encodings,face_encoding)
            print(matches)
	    #get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(face_distances)
            accurate=max(face_distances)*100
            if matches[best_match_index]:
                name = known_names[best_match_index]
    process_this_frame= not process_this_frame
    for (top, right, bottom, left) in face_locations:
	#creating a rectangle around face in frame		
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
	#putting image name on the top of rectangle 
        cv2.putText(frame, name, (left , top), font, 1.0, (255, 225, 0), 2)
        cv2.putText(frame,str(accurate), (10,50), font, 1.0, (255, 225, 0), 4)		
	#showing our frame	
        cv2.imshow("Live",frame)
    #press q to quit the camera 
    if cv2.waitKey(30) & 0xFF==ord('q'):
        break		
cam.release()
cv2.destroyAllWindows()

