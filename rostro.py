import cv2
import mediapipe as mp
import numpy as np
import math
import time
# audio
import playsound as ps


def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    return (d_A + d_B) / (2 * d_C)


def relacion_nariz(coordinates):
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
index_ojo_izquierdo = [33, 160, 158, 133, 153, 144]
index_ojo_derecho = [362, 385, 387, 263, 373, 380]
index_boca = [61, 37, 267, 291, 314, 84]
index_nariz = [36, 4]
valor_relacion_ojos_ref = 0.24

tiempo_sueno = 0
tiempo_sueno_real = 0
sueno = False
contador_sueno = 0
inicio_sueno = 0
final_sueno = 0
final = 0
muestra_sueno = 0
estado_sueno = True
# 2
valor_relacion_boca_ref = 0.60
tiempo_bostezo_real = 0
tiempo_bostezo = 0
bostezo = False
contador_bostezo = 0
inicio_bostezo = 0
final_bostezo = 0
muestra_bostezo = 0
estado_bostezo = True

# 3
valor_relacion_nariz_ref = 12
valor_relacion_nariz_ref2 = 40
tiempo_distraccion_real = 0
tiempo_distraccion = 0
distraccion = False
contador_distraccion = 0
inicio_distraccion = 0
final_distraccion = 0
muestra_distraccion = 0
estado_distraccion = True

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
         # Aplicar el espejo para evitar la inversión de la imagen
        frame = cv2.flip(frame, 1)

        # Aplicar mejoras en la imagen para una mejor detección de rasgos
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_gray = clahe.apply(frame_gray)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # Convertir la imagen a color nuevamente
        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

        # Continuar con la detección de rasgos faciales en 'frame_rgb'
        results = face_mesh.process(frame_rgb)

        height, width = frame.shape[:2]

        coordenadas_ojo_izquierda = []
        coordenadas_ojo_derecha = []
        coordenadas_boca = []
        coordenadas_nariz = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_ojo_izquierdo:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordenadas_ojo_izquierda.append([x, y])
                    # Dibujar puntos
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                for index in index_ojo_derecho:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordenadas_ojo_derecha.append([x, y])
                    # Dibujar puntos
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                for index in index_boca:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordenadas_boca.append([x, y])
                    # Dibujar puntos
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                for index in index_nariz:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordenadas_nariz.append([x, y])
                    # Dibujar puntos
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)

            ear_left_eye = eye_aspect_ratio(coordenadas_ojo_izquierda)
            ear_right_eye = eye_aspect_ratio(coordenadas_ojo_derecha)
            valor_relacion_ojos = (ear_left_eye + ear_right_eye)/2

            valor_relacion_boca = eye_aspect_ratio(coordenadas_boca)
            valor_relacion_nariz = relacion_nariz(coordenadas_nariz)

            #INICIO DE LAS CONDICIONES
            # Ojos cerrados
            if valor_relacion_ojos <= valor_relacion_ojos_ref and sueno == False:
                sueno = True
                estado_sueno = True
                inicio_sueno = time.time()

            elif valor_relacion_ojos > valor_relacion_ojos_ref and sueno == True:
                sueno = False
                final_sueno = time.time()
            
            tiempo_sueno = round(final_sueno - inicio_sueno, 0)

            if tiempo_sueno >= 3:
                contador_sueno += 1
                muestra_sueno = tiempo_sueno
                inicio_sueno = 0
                final_sueno = 0
                print("Ojos cerrados: " + str(tiempo_sueno) + " segundos")

            # Boca abierta
            if valor_relacion_boca > valor_relacion_boca_ref and bostezo == False:
                bostezo = True
                estado_bostezo = True

                inicio_bostezo = time.time()
                print(round(inicio_bostezo, 0))
            elif valor_relacion_boca < valor_relacion_boca_ref and bostezo == True:
                bostezo = False

                final_bostezo = time.time()
                print(round(final_bostezo - inicio_bostezo, 0))
                        
            # ------ Sonidos
            if tiempo_sueno >= 5 and round(final_bostezo - inicio_bostezo, 0) >= 1 or tiempo_sueno >= 5 and round(final_bostezo - inicio_bostezo, 0) >= 0:
                print('Manten los ojos abiertos 1!!!!!!!!!!!!!!!!!!!!!!!!')
		#Cambiar la dirección del sonido
                ps.playsound(
                        "C:/project/rostro/sonidos/sonido.wav")
            elif tiempo_sueno >= 5 and round(final_bostezo - inicio_bostezo, 0) >= 0:
                print('Manten los ojos abiertos 2!!!!!!!!!!!!!!!!!!!!!!!!')
		#Cambiar la dirección del sonido
                ps.playsound(
                        "C:/project/rostro/sonidos/sonido.wav")
            elif tiempo_sueno >= 3 and tiempo_sueno <= 5 and round(final_bostezo - inicio_bostezo, 0) >= 2:
                print('Manten los ojos abiertos 3!!!!!!!!!!!!!!!!!!!!!!!!')
		#Cambiar la dirección del sonido
                ps.playsound(
                        "C:/project/rostro/sonidos/sonido.wav")

        cv2.imshow("Processed Frame", frame_gray)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
