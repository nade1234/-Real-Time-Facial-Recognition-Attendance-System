# face_rec.py - Put this in your main project directory
# This file contains ONLY face recognition functionality, NO streamlit code

import numpy as np
import pandas as pd
import cv2
import redis
import os
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime

# Redis Connection
hostname = 'redis-15786.c15.us-east-1-2.ec2.redns.redis-cloud.com'
portnumber = 15786
password = '0KqdTtw5qQlQWUHE9eAJW7ps3y7Hjw7h'

r = redis.StrictRedis(
    host=hostname,
    port=portnumber,
    password=password
)

def retrieve_data(name):
    """
    Retrieve facial data from Redis and convert to DataFrame.
    Ultra robust: Handles malformed/empty/non-string keys.
    """
    retrieved = r.hgetall(name)
    valid_entries = {}

    for raw_key, raw_value in retrieved.items():
        try:
            name_key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
        except Exception as e:
            print(f"Erreur décodage clé: {e}")
            continue

        try:
            arr = np.frombuffer(raw_value, dtype=np.float32)
            if arr.shape[0] == 512:
                valid_entries[name_key] = arr
        except Exception as e:
            print(f"Erreur embedding pour {name_key}: {e}")

    retrive_series = pd.Series(valid_entries)
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'facial_features']

    # Construire Name et Role de façon robuste
    names, roles = [], []
    for val in retrive_df['name_role'].astype(str):
        parts = val.split('@')
        if len(parts) == 2 and all(parts):  # nom@role non vide
            names.append(parts[0])
            roles.append(parts[1])
        else:
            names.append("Unknown")
            roles.append("Unknown")

    retrive_df['Name'] = names
    retrive_df['Role'] = roles

    # Garde seulement les vrais noms/roles (optionnel, sinon tout sera listé)
    retrive_df = retrive_df[retrive_df['Name'] != "Unknown"]

    return retrive_df




# Initialize InsightFace
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_role=['Name', 'Role'], thresh=0.5):
    """
    Cosine similarity search
    """
    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def savelogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if encoded_data:
            r.lpush('attendance:logs', *encoded_data)
            print('✅ Attendance logs saved to Redis.')

        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column,
                        name_role=['Name', 'Role'], thresh=0.5):
        current_time = str(datetime.now())
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                           feature_column,
                                                           test_vector=embeddings,
                                                           name_role=name_role,
                                                           thresh=thresh)
            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

            cv2.putText(test_copy, person_name, (x1, y1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2 + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy

class RegistrationForm:
    def __init__(self):
        self.sample = 0
        self.embeddings_list = []  # Store embeddings in memory

    def reset(self):
        self.sample = 0
        self.embeddings_list = []  # Clear stored embeddings
        # Remove embedding file if it exists
        if os.path.exists('face_embedding.txt'):
            os.remove('face_embedding.txt')

    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None

        if not results:
            print("⚠️ No face detected.")
            return frame, None

        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

            embeddings = res['embedding']
            if embeddings is not None and embeddings.shape == (512,):
                # Store embedding in memory instead of writing line by line
                self.embeddings_list.append(embeddings)
            else:
                print("⚠️ Invalid embedding shape.")

        return frame, embeddings

    def save_data_in_redis_db(self, name, role):
        if name is None or name.strip() == '':
            return 'name_false'

        key = f'{name}@{role}'
        
        # Check if we have collected any embeddings
        if len(self.embeddings_list) == 0:
            return 'embedding_false'
            
        # Convert list of embeddings to numpy array
        x_array = np.array(self.embeddings_list, dtype=np.float32)
        
        # Save all embeddings to file at once (for backup/debugging)
        with open('face_embedding.txt', 'wb') as f:
            np.savetxt(f, x_array)
            
        # Calculate mean of embeddings
        x_mean = x_array.mean(axis=0)
        x_mean_bytes = x_mean.tobytes()
        
        # Save to Redis
        r.hset(name='academy:register', key=key, value=x_mean_bytes)
        
        # Clear saved data
        self.reset()

        print("✅ Data saved to Redis.")
        return True