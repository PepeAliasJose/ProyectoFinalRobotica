##################### importamos las librerias ####################
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import time
import telebot
import RPi.GPIO as GPIO
###################### variables para que el programa pueda arrancar ####################
fb=0

Y=0
X=0
clase=0
p=0

TOKEN=''
tb = telebot.TeleBot(TOKEN)
chatid=''
# Load the COCO Label Map
elapsed = []
####################### categorias de deteccion ########################
category_index = {
    1: {'id': 1, 'name': 'persona'},
    2: {'id': 2, 'name': 'rueda'},
    3: {'id': 3, 'name': 'coche_bien'}
}
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT, initial=GPIO.LOW)#rueda cilindro
GPIO.setup(12, GPIO.OUT, initial=GPIO.LOW)#fallo persona
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)#rueda en cinta
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)#rueda en rampa
############################################## funcion para cargar el modelo ##################################################3
def loadTensorflowModel():
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(
        '/home/pepe/Desktop/proyecto/exported-model/saved_model/')#ruta con la carpeta donde esta el modelo 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ' + str(elapsed_time) + 's')
    return detect_fn


detect_fn= loadTensorflowModel() #cargar el modelo


############################################## Empieza el codigo #########################################################################
########################################################################################
dispW=400 #ancho y alto en pixeles
dispH=300
flip=0 #gira la orientacion de la camara
########################################################################################
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3000, height=2400, format=NV12, framerate=3/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# parametros de la camara y formato de video
cam=cv2.VideoCapture(camSet) #cam set es la camara conectada al mipi-csi-2
cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 0)
########################################################################################

while True:
    _,frame = cam.read()
    image_rgb_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#copiamos el frame en otra variable
    inicio=time.time()#variable para calcular los fps
    ###################################
    input_tensor = np.expand_dims(image_rgb_np, 0)#convertimos la imagen en un tensor y no le añadimos ninguna dimension más
    detections = detect_fn(input_tensor)#metemos la imagen en el modelo cargado

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_rgb_np.copy()#copiamos la imagen y la metemos en otra variable importante el .copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(#pintamos las deteciones en la imagen segun unos parametros
        image_np_with_detections,#imagen
        detections['detection_boxes'][0].numpy(),#tensor de cajas de deteccion convertido a array
        detections['detection_classes'][0].numpy().astype(np.int32),#tensor de clases convertido a una lista de enteros int32
        detections['detection_scores'][0].numpy(),#tensor de procentajes convertido a array
        category_index,#categorias establecidas arriba
        use_normalized_coordinates=True,#coordenadas normalizadas para las cajas
        max_boxes_to_draw=2,#maximas de cajas que pintar o maximas detecciones
        min_score_thresh=.55,#porcentage de confianza con la deteccion para pintarla
        agnostic_mode=False)
    
    
    print("Y:",int((detections['detection_boxes'][0][0][0]*300)))
    print("X:",int((detections['detection_boxes'][0][0][1]*400)))
    #print("Clase:", int(detections['detection_classes'][0][0].numpy().astype(np.int32)))
    #print("Probabilidad:",detections['detection_scores'][0][0].numpy())
    
    Y = int((detections['detection_boxes'][0][0][0]*300)) #convertimos el valor concreto del array del tensor en un numero entero y se multiplica para tener la coordenada en pixeles
    X = int((detections['detection_boxes'][0][0][1]*400)) #convertimos el valor concreto del array del tensor en un numero entero y se multiplica para tener la coordenada en pixeles
    clase = int(detections['detection_classes'][0][0].numpy().astype(np.int32))# convertir valor concreto del array en entero para saber que clase se detecta
    p = detections['detection_scores'][0][0].numpy()#coger el valor concreto del array de detecciones 
    
    #return image_np_with_detections
    image_rgb_np_with_detections=cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    print("P:",p)
    print("Clase:",clase)
    print("Flanco de brazo:",fb)

    if clase==1 and p>=0.65 and fb==0: #if para mandar una foto si detecta un brazo y parar la instalacion
        cv2.imwrite('/home/pepe/Desktop/proyecto/foto_brazo.jpg',image_rgb_np_with_detections)
        foto = open('/home/pepe/Desktop/proyecto/foto_brazo.jpg','rb')
        tb.send_message(chatid,'Hay alguien tocando donde no debe cuando no debe UwU')
        tb.send_photo(chatid,foto)
        fb=1
        GPIO.output(12,1)
        print("Hehe que haces?")
    if clase!=1 or p<=0.45:
        fb=0
        GPIO.output(12,0)
    ###################################
    if clase==2 and p>=0.55 and Y<40:
        GPIO.output(18,1)
        print("Hay una rueda en la cinta")
    if clase==2 and p>=0.55 and X>=136 and Y<40:
        GPIO.output(23,1)
        GPIO.output(18,0)
        print("Hay una rueda en todo el cilindro")
    if clase==2 and p>=0.55 and Y>120:
        GPIO.output(13,1)
        GPIO.output(23,0)
        print("Ven a por la rueda bracito de fanuc")
    if clase!=2 or p<=0.5:
        GPIO.output(13,0)
        GPIO.output(18,0)
        GPIO.output(23,0)



    final=time.time()
    pas=final-inicio
    fps=1/pas #calcular los fps
    print("FPS:",fps)
    cv2.imshow('GLaDOS',cv2.resize(image_rgb_np_with_detections,(600,400)))

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()