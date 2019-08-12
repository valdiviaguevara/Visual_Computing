#conda create -n tensorflow_cpu pip python=3.6
#conda install scipy
#activate tensorflow_cpu ---------------Serve para activar otensorflow_cpu
#INSTALANDO TENSORFLOW
#pip install --ignore-installed --upgrade tensorflow==1.13.1
#INSTALANDO KERAS
#conda install -c anaconda keras
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
#------------------------------

#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
#------------------------------
#variables
num_classes = 7 #Bravo, Disgustado, Medo, Feliz, Triste, Sorpreso, Neutro
batch_size = 256
epochs = 5
#------------------------------
#Base de dados de Reconhecimento de Sentimentos
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data/fer2013/fer2013.csv
with open("Dados/Base_Sentimento.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("Numero de Instancias: ",num_of_instances)
print("Longitud de Instancias: ",len(lines[1].split(",")[1].split(" ")))

#------------------------------
#Inicializando o Traino e Teste
x_train, y_train, x_test, y_test = [], [], [], []

#------------------------------
#Transfirindo o conjunto de dados de traino e teste
for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
    	print("",end="")

#------------------------------
#Conjunto de dados de transformacion para treino e teste 
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #Normalizando entradas emtre [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#------------------------------
#Construindo estrutura CNN
model = Sequential()

#1er camada de comvolucion
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd camada de comvolucion
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd camada de comvolucion
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#Redes Neurais completamente conetadas
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
#------------------------------
#proceso batch
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

#------------------------------

model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)

#------------------------------

fit = True

if fit == True:
	#model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
	model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #Treinar para um seleccionado aleatoriamente
else:
	model.load_weights('/Dados/facial_expression_model_weights.h5') #lendo pesos
#Funcion para graficar os histogramas para cada sentimento predicho
def emotion_analysis(emotions):
    objects = ('bravo', 'disgustado', 'medo', 'feliz', 'triste', 'sorpreso', 'neutro')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Porcentagem')
    plt.title('Sentimento')
    
    plt.show()
#------------------------------

monitor_testset_results = False

if monitor_testset_results == True:
	#fazer previciones para o conjunto de teste
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1

#------------------------------
#Fazendo previciones para imagenes personalizadas fora do conjunto de teste

img = image.load_img("Imagenes/TT_4.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()
#