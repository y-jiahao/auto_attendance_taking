import tensorflow as tf
from deepface import DeepFace
import os
from tqdm import tqdm
import pickle
import pandas as pd
import cv2
import math

# generate face_loc.csv
def generate_face_loc(eval_path = 'evaluation_images'):
    print('\n[PROCESS] Generating face_loc.csv.')
    print('\n[STATUS] Checking dependencies...')

    # check if GPU is available
    if len(tf.config.list_physical_devices('GPU')):
        print('[*] GPU device found.')
    else:
        print('[*] No GPU device found. Using CPU instead.')

    dirname = os.path.dirname(__file__)
    evalfullpath = os.path.join(dirname, eval_path)

    # check if evaluation_images folder exists
    if not os.path.isdir(evalfullpath):
        print('[ERROR] No '+eval_path+' folder found.')
        return
    else:
        print('[*] '+eval_path+' folder found.')

    # get all image files in the evaluation_images folder
    files = [i for i in os.listdir(evalfullpath) if i.endswith('.jpg')]
    print('[*] '+str(len(files))+' images found in '+eval_path+'.')

    # build model
    af_build_model = DeepFace.modeling.build_model("ArcFace")

    face_loc = []

    print('\n[STATUS] Generating face_loc.csv...')

    for i in range(len(files)):
        file = files[i]
        file_path = os.path.join(evalfullpath, file)

        # extract face from image
        source_objs = DeepFace.detection.extract_faces(
            img_path=file_path,
            target_size=af_build_model.input_shape,
            detector_backend='retinaface',
            grayscale=False,
            enforce_detection=True,
            align=True
        )

        # for each face extracted, get the facial area
        for source_obj in source_objs:
            source_region = source_obj["facial_area"]

            x = source_region['x']
            y = source_region['y']
            w = source_region['w']
            h = source_region['h']

            face_loc.append([file, x, y, w, h, ""])

    # save face_loc.csv
    face_loc_df = pd.DataFrame(face_loc, columns=['file', 'x', 'y', 'w', 'h', 'identity'])
    face_loc_df.to_csv(evalfullpath+'/face_loc.csv', index=False)

    print('\n[STATUS] face_loc.csv generated successfully.')
    print('[!] Identity column is left empty. Please fill in the identities accordingly.')