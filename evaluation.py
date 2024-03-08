import tensorflow as tf
from deepface import DeepFace
import os
from tqdm import tqdm
import pickle
import pandas as pd
import cv2
import math

import utilities as ut

# evaluate labelled images for attendance taking
def evaluation(ensemble, db_path= 'database_images', eval_path = 'evaluation_images', result_path = 'evaluation_images/result_images', face_loc_path = 'evaluation_images/face_loc.csv'):
    print('\n[PROCESS] Evaluating images for attendance taking.')
    print('\n[STATUS] Checking dependencies...')

    # check if GPU is available
    if len(tf.config.list_physical_devices('GPU')):
        print('[*] GPU device found.')
    else:
        print('[*] No GPU device found. Using CPU instead.')

    dirname = os.path.dirname(__file__)

    rep_ensemble_fn512 = os.path.join(dirname, db_path+'/representations_ensemble_facenet512.pkl')
    rep_ensemble_af = os.path.join(dirname, db_path+'/representations_ensemble_arcface.pkl')
    rep_fn512 = os.path.join(dirname, db_path+'/representations_facenet512.pkl')
    rep_af = os.path.join(dirname, db_path+'/representations_arcface.pkl')

    # check if embeddings files exist
    if ensemble:
        if not os.path.isfile(rep_ensemble_fn512):
            print('[ERROR] No representations_ensemble_facenet512.pkl file found in '+db_path+'.')
            return
        if not os.path.isfile(rep_ensemble_af):
            print('[ERROR] No representations_ensemble_arcface.pkl file found in '+db_path+'.')
            return
    else:
        if not os.path.isfile(rep_fn512):
            print('[ERROR] No representations_facenet512.pkl file found in '+db_path+'.')
            return
        if not os.path.isfile(rep_af):
            print('[ERROR] No representations_arcface.pkl file found in '+db_path+'.')
            return
        
    evalfullpath = os.path.join(dirname, eval_path)
    resultfullpath = os.path.join(dirname, result_path)
    faclocfullpath = os.path.join(dirname, face_loc_path)

    # check if evaluation_images folder exists
    if not os.path.isdir(evalfullpath):
        print('[ERROR] No '+eval_path+' folder found.')
        return
    else:
        print('[*] '+eval_path+' folder found.')
    
    # check if evaluation_images/result_images folder exists
    if not os.path.isdir(resultfullpath):
        print('[ERROR] No '+result_path+' folder found.')
        return
    else:
        print('[*] '+result_path+' folder found.')

    # check if face_loc.csv file exists
    if not os.path.isfile(faclocfullpath):
        print('[ERROR] No '+face_loc_path+' file found.')
        return
    else:
        print('[*] '+face_loc_path+' file found.')
    
    # build AF model
    af_model_name = 'ArcFace'
    af_build_model = DeepFace.modeling.build_model(af_model_name)
    af_metric = 'cosine'

    # build FN512 model
    fn_model_name = 'Facenet512'
    fn_build_model = DeepFace.modeling.build_model(fn_model_name)
    fn_metric = 'euclidean_l2'

    # load embeddings
    if ensemble:
        with open(rep_ensemble_af, "rb") as f:
            af_embeddings = pickle.load(f)

        with open(rep_ensemble_fn512, "rb") as f:
            fn_embeddings = pickle.load(f)
    else:
        with open(rep_af, "rb") as f:
            af_embeddings = pickle.load(f)

        with open(rep_fn512, "rb") as f:
            fn_embeddings = pickle.load(f)

    af_df = pd.DataFrame(af_embeddings, columns=["identity", f"{af_model_name}_representation"])
    fn_df = pd.DataFrame(fn_embeddings, columns=["identity", f"{fn_model_name}_representation"])

    combined_df = pd.merge(af_df, fn_df, on='identity')
    print('[*] '+str(len(combined_df))+' embeddings found in the database.')

    # fetch all images in the evaluation_images folder
    files = [i for i in os.listdir(evalfullpath) if (i.endswith('.jpg') or i.endswith('.png'))]
    print('[*] '+str(len(files))+' images found for evaluation.')

    face_loc_df = pd.read_csv(os.path.join(dirname, eval_path+'/face_loc.csv'))

    # check if face_loc.csv contains all the evaluation images
    for i in range(len(files)):
        file = files[i]
        if file not in face_loc_df['file'].values:
            print(f'[ERROR] No face_loc entry for {file}.')
            return

    correct = 0
    total = 0
    unknown = 0

    # create log file
    if ensemble:
        output_file_path = evalfullpath+"/log_evaluationoutputs_ensemble.txt"
    else:
        output_file_path = evalfullpath+"/log_evaluationoutputs.txt"
    # clear log file
    with open(output_file_path, 'w') as f:
        f.write('')

    temp_path = resultfullpath+'/temp.jpg'

    print('\n[STATUS] Evaluating images...')
    print('[!] Intializing the model for the first time may take a while.\n')

    for i in tqdm(range(len(files))):
        file_correct = 0
        file_total = 0
        file_unknown = 0

        file = files[i]
        file_path = os.path.join(evalfullpath, file)

        resp_obj = {}

        file_image = cv2.imread(file_path)

        for model_name, metric, target_size in [[af_model_name, af_metric, af_build_model.input_shape], [fn_model_name, fn_metric, fn_build_model.input_shape]]:
            # extract faces from image
            source_objs = DeepFace.detection.extract_faces(
                img_path=file_path,
                target_size=target_size,
                detector_backend='retinaface',
                grayscale=False,
                enforce_detection=True,
                align=True
            )

            for source_obj in source_objs:
                source_region = source_obj["facial_area"]
                margin = 20
                try:
                    while True:
                        # crop face from image and upscale it to 500px width
                        crop_img = file_image[max(source_region['y']-margin,0):min(source_region['y']+source_region['h']+margin, file_image.shape[0]), max(source_region['x']-margin,0):min(source_region['x']+source_region['w']+margin, file_image.shape[1])]
                        width = 500
                        scale = width / crop_img.shape[1]
                        height = int(crop_img.shape[0] * scale)
                        crop_img = cv2.resize(crop_img, (width, height))
                        cv2.imwrite(temp_path, crop_img)

                        # extract face from upscaled image
                        temp_source_objs = DeepFace.detection.extract_faces(
                            img_path=temp_path,
                            target_size=target_size,
                            detector_backend='retinaface',
                            grayscale=False,
                            enforce_detection=True,
                            align=True
                        )
                        if len(temp_source_objs) == 1:
                            source_img = temp_source_objs[0]["face"]
                            break
                        else:
                            margin -= 5
                except:
                    # margin = 10
                    # crop_img = file_image[max(source_region['y']-margin,0):min(source_region['y']+source_region['h']+margin, file_image.shape[0]), max(source_region['x']-margin,0):min(source_region['x']+source_region['w']+margin, file_image.shape[1])]
                    # width = 200
                    # scale = width / crop_img.shape[1]
                    # height = int(crop_img.shape[0] * scale)
                    # crop_img = cv2.resize(crop_img, (width, height))
                    # temp_path = f'{db_path}/temp.jpg'
                    # cv2.imwrite(temp_path, crop_img)

                    # temp_source_objs = DeepFace.detection.extract_faces(
                    #     img_path=temp_path,
                    #     target_size=target_size,
                    #     detector_backend='skip',
                    #     grayscale=False,
                    #     enforce_detection=True,
                    #     align=True
                    # )

                    source_img = source_obj["face"]      

                # get target embedding for face
                target_embedding_obj = DeepFace.represent(
                    img_path=source_img,
                    model_name=model_name,
                    enforce_detection=True,
                    detector_backend="skip",
                    align=True,
                    normalization='base',
                )

                target_representation = target_embedding_obj[0]["embedding"]
                result_df = combined_df.copy()  # df will be filtered in each img
                result_df["source_x"] = source_region["x"]
                result_df["source_y"] = source_region["y"]
                result_df["source_w"] = source_region["w"]
                result_df["source_h"] = source_region["h"]

                distances = []
                # calculate distance between target and all embeddings in the database
                for _, instance in combined_df.iterrows():
                    source_representation = instance[f"{model_name}_representation"]

                    if metric == "cosine":
                        distance = DeepFace.verification.find_cosine_distance(
                            source_representation, target_representation
                        )
                    elif metric == "euclidean":
                        distance = DeepFace.verification.find_euclidean_distance(
                            source_representation, target_representation
                        )
                    elif metric == "euclidean_l2":
                        distance = DeepFace.verification.find_euclidean_distance(
                            DeepFace.verification.l2_normalize(source_representation),
                            DeepFace.verification.l2_normalize(target_representation),
                        )
                    
                    distances.append(distance)

                result_df[f"{model_name}_distance"] = distances
                result_df = result_df.drop(columns=[f"{af_model_name}_representation"])
                result_df = result_df.drop(columns=[f"{fn_model_name}_representation"])
                result_df = result_df.sort_values(by=[f"{model_name}_distance"], ascending=True).reset_index(drop=True)

                # merge distances from both models
                if (source_region["x"], source_region["y"]) in resp_obj:
                    existing_df = pd.DataFrame(resp_obj[(source_region["x"], source_region["y"])])
                    merged_df = pd.merge(existing_df, result_df, on=['identity', 'source_x', 'source_y', 'source_w', 'source_h'])
                    if source_region["x"] >= 30 or source_region["y"] >= 30:
                        merged_df["combined_distance"] = 0.83*merged_df[f"{af_model_name}_distance"] + 0.17*merged_df[f"{fn_model_name}_distance"]
                    else:
                        merged_df["combined_distance"] = 0.83*merged_df[f"{af_model_name}_distance"] + 0.17*merged_df[f"{fn_model_name}_distance"]
                    merged_df = merged_df.sort_values(by=["combined_distance"], ascending=True).reset_index(drop=True)
                    resp_obj[(source_region["x"], source_region["y"])] = merged_df
                else:
                    resp_obj[(source_region["x"], source_region["y"])] = result_df

        image = cv2.imread(file_path)

        if ensemble:  
            m_file_path = os.path.join(resultfullpath, file[:-4]+'_eval_ensemble.jpg')
        else:
            m_file_path = os.path.join(resultfullpath, file[:-4]+'_eval.jpg')

        # annotate evaluation image with recognition results and save
        for face in resp_obj.values():
            face_df = pd.DataFrame(face)
            top_result = face_df.iloc[0]
            if type(top_result['identity']) is float and math.isnan(top_result['identity']):
                continue
            
            x = top_result["source_x"]
            y = top_result["source_y"]
            result = top_result['identity'].split('_')[0]
            truth = face_loc_df[(face_loc_df['file'] == file) & (face_loc_df['x'] == x) & (face_loc_df['y'] == y)]['identity'].values[0]
            file_total += 1
            if truth == result:
                file_correct += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            if truth == 'Unknown':
                file_unknown += 1
            distance = round(float(top_result['combined_distance']),2)
            cv2.rectangle(image, (x, y), (x+top_result['source_w'], y+top_result['source_h']), color, 2)
            ut.draw_text(image, f'{result[:10]}', pos=(x, y-18))
            ut.draw_text(image, f'{distance}', pos=(x, y-5))
            
            cv2.imwrite(m_file_path, image)

        total += file_total
        correct += file_correct
        unknown += file_unknown

        # write image evaluation results to log file
        with open(output_file_path, 'a') as f:
            f.write(f'{file}\n')
            f.write(f'Accuracy: {file_correct}/{file_total} ({round(file_correct/file_total*100,2)}), Unknown: {file_unknown}\n\n')

    # write total evaluation results to log file
    with open(output_file_path, 'a') as f:
        f.write(f'Total accuracy: {correct}/{total} ({round(correct/total*100,2)}), Unknown: {unknown}\n\n')

    if os.path.exists(temp_path):
        os.remove(temp_path)

    print('\n[STATUS] '+str(len(files))+' images evaluated successfully.')

    if ensemble:
        print('[*] Evaluation outputs saved in file log_recognitionoutputs_ensemble.txt')
    else:
        print('[*] Evaluation outputs saved in file log_recognitionoutputs.txt')

    print('[*] Evaluation images saved in '+result_path+' folder.')