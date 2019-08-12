import numpy as np
import cv2
from keras.preprocessing import image
import time
#-----------------------------
#Inicializando o opencv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#-----------------------------
#Inicializando o reconhecimento de expressao do rostro
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #cargando os pesos

#-----------------------------

emotions = ('BRAVO', 'DISGUSTADO', 'MEDO', 'FELIZ', 'TRISTE', 'SURPRESO', 'NEUTRO')

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#print(faces) #lugar para detetar rostros

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #dibujando um retangulo para a imagem principal
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #Construindo a cara detectada
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transformando a escala gris
		detected_face = cv2.resize(detected_face, (48, 48)) #redimencionando para 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #os pixels sao em escaa de [0, 255]. normalizando todos os pixels na escala de [0, 1]
		
		predictions = model.predict(img_pixels) #Guardando asprobabilidades das 7 expresoes
		
		#Encontrando o indice MAximo do  array 0: Bravo, 1:disgustado, 2:Medo, 3:Feliz, 4:Triste, 5:sorpreso, 6:neutro
		max_index = np.argmax(np.round(100*predictions[0],decimals =1))
		emotion0 = emotions[0]
		emotion1 = emotions[1]
		emotion2 = emotions[2]
		emotion3 = emotions[3]
		emotion4 = emotions[4]
		emotion5 = emotions[5]
		emotion6 = emotions[6]
		
		#escrevendo as legendas das emocoes do lado do retangulo
		cv2.putText(img,"| "+emotion0 +"      = "+ str(np.round(100*predictions[0],decimals =1)[0])+" %", (int(x)+200, int(y)), cv2.FONT_HERSHEY_COMPLEX   , 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion1 +"= "+ str(np.round(100*predictions[0],decimals =1)[1])+" %", (int(x)+200, int(y)+20), cv2.FONT_HERSHEY_COMPLEX  , 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion2 +"       = "+ str(np.round(100*predictions[0],decimals =1)[2])+" %", (int(x)+200, int(y)+40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion3 +"       = "+ str(np.round(100*predictions[0],decimals =1)[3])+" %", (int(x)+200, int(y)+60), cv2.FONT_HERSHEY_COMPLEX , 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion4 +"      = "+ str(np.round(100*predictions[0],decimals =1)[4])+" %", (int(x)+200, int(y)+80), cv2.FONT_HERSHEY_COMPLEX  , 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion5 +"  = "+ str(np.round(100*predictions[0],decimals =1)[5])+" %", (int(x)+200, int(y)+100), cv2.FONT_HERSHEY_COMPLEX , 0.5, (0,0,0), 1)
		cv2.putText(img,"| "+emotion6 +"     = "+ str(np.round(100*predictions[0],decimals =1)[6])+" %", (int(x)+200, int(y)+120), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
		#time.sleep(0.1)
		#-------------------------

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #Presionando para sair
		break

#Parando os procesos de open cv		
cap.release()
cv2.destroyAllWindows()