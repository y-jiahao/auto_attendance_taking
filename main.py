import generate_embeddings as ge
import evaluation as eval
import generate_face_loc as gfl
import prediction as pred

if __name__ == '__main__':

    while(True):
        print('\n=============== Automated Attendance Tacking System ===============')
        print('\n[!] Generate embeddings first if there are changes to the image files \nin the database_images folder before evaluating or predicting images.')
        print('1. Generate normal embeddings')
        print('2. Generate ensemble embeddings')
        print('\n[!] Generate face_loc.csv first and fill in the identities accordingly \nbefore evaluating recognition.')
        print('3. Generate face_loc.csv')
        print('4. Evaluate recognition on labelled images using normal embeddings')
        print('5. Evaluate recognition on labelled images using ensemble embeddings')
        print('\n6. Predict recognition on unlabelled images using normal embeddings')
        print('7. Predict recognition on unlabelled images using ensemble embeddings')
        print('\n8. Exit')

        choice = input('\nEnter your choice: ')

        if choice == '1':
            ge.generate_normal_embeddings('Facenet512')
            ge.generate_normal_embeddings('ArcFace')
        elif choice == '2':
            ge.generate_ensemble_embeddings('Facenet512')
            ge.generate_ensemble_embeddings('ArcFace')
        elif choice == '3':
            gfl.generate_face_loc()
        elif choice == '4':
            eval.evaluation(False)
        elif choice == '5':
            eval.evaluation(True)
        elif choice == '6':
            pred.prediction(False)
        elif choice == '7':
            pred.prediction(True)
        elif choice == '8':
            exit()
        else:
            print('Invalid choice. Please try again.')