# Automated Attendance Taking System

Developed for my Final Year Project on Large Group Facial Recognition for Attendance Taking

### Model Backbone:
| Stage | Models |
| ----- | ------ |
| Facial Detection | RetinaFace |
| Facial Recognition | FaceNet512 with Euclidean L2 similarity + ArcFace with Cosine similarity |

*Models and methods are adapted from the [InsightFace](https://github.com/deepinsight/insightface) framework*


### How to use:
1. Install dependencies:

    >Optional: Use a virtual environment via python venv or conda create (recommended)
    ```
    pip install -r requirements
    ```
2. Create folders:

    >Create the following folders in the specified order and locations

    - `database_images`: 
        - Create in the **main directory**
        - Place all photos of individuals in this folder (.jpg or .png)
        - Each photo should be labelled in the format of *<identity_name>_<photo_number>*
            - Eg. Tan Ah Kow_01, Tan Ah Kow_02, Lily Koh_01, Lily Koh_02

    - `evaluation_images`:
        - Create in the **main directory**
        - Place all photos of large group images to be evaluated in this folder (.jpg or .png)
        - `result_images`: 
            - Create in the **evaluation_images folder**

    - `prediction_images`: 
        - Create in the **main directory**
        - Place all photos of large group images to be predicted in this folder (.jpg or .png)
        - `result_images`: 
            - Create in the **prediction_images folder**

3. Run `main.py`:

    >Follow on screen instructions to navigate the program

    >Notes:
    >- Embeddings must be generated using option 1 or 2 before evaluation or prediction
    >- `face_loc.csv` must be generated using option 3 before evaluation (`face_loc.csv` will be the file containing the truth labels for the evaluation images)
    >- Upon generation of `face_loc.csv`, the identity column have to be manually filled up before evaluation or it will result in 0% accuracy
    >- Evaluation is used for labelled images where accuracy metrics will be the output
    >- Prediction is used for unlabelled images where the identities identified will be the output
