import tensorflow as tf
from deepface import DeepFace
import os
from tqdm import tqdm
import pickle
import time
import numpy as np

def generate_normal_embeddings(model_name, db_path = 'database_images'):
    print('\n[PROCESS] Generating normal embeddings with '+model_name+'.')
    print('\n[STATUS] Checking dependencies...')
    
    if len(tf.config.list_physical_devices('GPU')):
        print('[*] GPU device found.')
    else:
        print('[*] No GPU device found. Using CPU instead.')

    if model_name != 'Facenet512' and model_name != 'ArcFace':
        print(f'[ERROR] Model "{model_name}" not found.')
        return
    
    dirname = os.path.dirname(__file__)
    fullpath = os.path.join(dirname, db_path)

    if not os.path.isdir(fullpath):
        print('[ERROR] No '+db_path+' folder found.')
        return
    else:
        print('[*] '+db_path+' folder found.')
    
    build_model = DeepFace.modeling.build_model(model_name)
    
    output_file_path = fullpath+"/log_normal_embeddings_"+model_name.lower()+".txt"

    with open(output_file_path, 'w') as f:
        f.write('')

    files = [i for i in os.listdir(fullpath) if (i.endswith('.jpg') or i.endswith('.png'))]

    print('[*] '+str(len(files))+' images found in the database.')

    print('\n[STATUS] Generating representations...')
    print('[!] Intializing the model for the first time may take a while.\n')

    embeddings = []
    errors = []
    time_start = time.time()

    for i in tqdm(range(0, len(files))):
        f = files[i]
        img_path = os.path.join(fullpath, f)
        try:
            img_objs = DeepFace.detection.extract_faces(
                img_path=img_path,
                target_size=build_model.input_shape,
                detector_backend='retinaface',
                grayscale=False,
                enforce_detection=True,
                align=True,
            )

            for img_obj in img_objs:
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                embedding_obj = DeepFace.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=True,
                    detector_backend="skip",
                    align=True,
                    normalization='base',
                )
                embedding = embedding_obj[0]["embedding"]
                embeddings.append([f.split('.')[0], embedding])
        except:
            errors.append(f)
    time_end = time.time()

    with open(f"{fullpath}/representations_{model_name.lower()}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print('\n[STATUS] '+str(len(files))+' embeddings generated successfully.')

    with open(output_file_path, 'a') as f:
        # f.write(f'{model_name}\n')
        f.write(f'No. errors: {len(errors)}\n')
        f.write('Error identities:\n' +"\n".join(errors)+'\n')
        f.write(f'\nTime taken: {time_end-time_start}')

    if len(errors):
        print('[*] '+str(len(errors))+' errors found. Check the file log_normal_embeddings_'+model_name.lower()+'.txt for more details.')


def generate_ensemble_embeddings(model_name, db_path = 'database_images'):
    print('\n[PROCESS] Generating ensemble embeddings with '+model_name+'.')
    print('\n[STATUS] Checking dependencies...')
    
    if len(tf.config.list_physical_devices('GPU')):
        print('[*] GPU device found.')
    else:
        print('[*] No GPU device found. Using CPU instead.')

    if model_name != 'Facenet512' and model_name != 'ArcFace':
        print(f'[ERROR] Model "{model_name}" not found.')
        return
    
    dirname = os.path.dirname(__file__)
    fullpath = os.path.join(dirname, db_path)

    if not os.path.isdir(fullpath):
        print('[ERROR] No '+db_path+' folder found.')
        return
    else:
        print('[*] '+db_path+' folder found.')
    
    build_model = DeepFace.modeling.build_model(model_name)
    
    output_file_path = fullpath+"/log_ensemble_embeddings_"+model_name.lower()+".txt"

    with open(output_file_path, 'w') as f:
        f.write('')

    files = [i for i in os.listdir(fullpath) if (i.endswith('.jpg') or i.endswith('.png'))]

    files_grp = {}
    for f in files:
        image_num = f.split('_')[0]
        if image_num in files_grp:
            files_grp[image_num].append(f)
        else:
            files_grp[image_num] = [f]

    print('[*] '+str(len(files))+' images/'+str(len(files_grp))+' identities found in the database.')

    print('\n[STATUS] Generating representations...')
    print('[!] Intializing the model for the first time may take a while.\n')

    embeddings = []
    errors = []
    time_start = time.time()

    for i in tqdm(range(0, len(files_grp))):
        f_list = list(files_grp.values())[i]
        identity_embeddings = []
        error = 0
        identity = f_list[0].split('_')[0]
        for f in f_list:
            img_path = os.path.join(fullpath, f)
            try:
                img_objs = DeepFace.detection.extract_faces(
                    img_path=img_path,
                    target_size=build_model.input_shape,
                    detector_backend='retinaface',
                    grayscale=False,
                    enforce_detection=True,
                    align=True,
                )

                for img_obj in img_objs:
                    img_content = img_obj["face"]
                    img_region = img_obj["facial_area"]
                    embedding_obj = DeepFace.represent(
                        img_path=img_content,
                        model_name=model_name,
                        enforce_detection=True,
                        detector_backend="skip",
                        align=True,
                        normalization='base',
                    )
                    embedding = embedding_obj[0]["embedding"]
                    identity_embeddings.append(embedding)
            except:
                error += 1
        if error < 4:
            combined_embeddings = np.mean(identity_embeddings, axis=0)
            embeddings.append([identity, combined_embeddings.tolist()])
        else:
            errors.append(identity)
    time_end = time.time()

    with open(f"{fullpath}/representations_ensemble_{model_name.lower()}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print('\n[STATUS] '+str(len(files_grp))+' embeddings generated successfully.')

    with open(output_file_path, 'a') as f:
        # f.write(f'{model_name}\n')
        f.write(f'No. errors: {len(errors)}\n')
        f.write('Error identities:\n' +"\n".join(errors)+'\n')
        f.write(f'\nTime taken: {time_end-time_start}')

    if len(errors):
        print('[*] '+str(len(errors))+' errors found. Check the file log_normal_embeddings_'+model_name.lower()+'.txt for more details.')