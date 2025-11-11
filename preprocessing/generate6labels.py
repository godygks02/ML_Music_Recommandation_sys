import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# 4 labels => 6 labels

# train data path
FILE_NAME = './data/train_set_with_metadata.csv'

# output file path
OUTPUT_FILE_NAME = 'labeled_dataset.csv'  

# select features to use clustering
feature_cols = ['duration_ms', 'danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']

OPTIMAL_K = 6  # K=6

label_mapping = {
    0: 'Happy',
    1: 'Energetic',
    2: 'Angry',
    3: 'Romantic',
    4: 'Calm',
    5: 'Sad' 
}

# data load
if not os.path.exists(FILE_NAME):
    print(f"error: '{FILE_NAME}' cannot find file.")
else:
    print(f"'{FILE_NAME}' loaded.")
    df = pd.read_csv(FILE_NAME)
    
    # drop null data
    df_clean = df.dropna(subset=feature_cols)

    # feature scaling
    X = df_clean[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA(n=2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled) 

    # K-means
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    # add cluster labels
    df_labeled = df_clean.copy()
    df_labeled['cluster_id'] = cluster_labels

    # label mapping
    df_labeled['new_label'] = df_labeled['cluster_id'].map(label_mapping)
    columns = ['track_name','album_name','artists','duration_ms','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','uri','new_label']
    df_final = df_labeled[columns]
    print(df_final.head(5))
    # save to csv
    try:
        df_labeled.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
        print(f"\nsuccess: '{OUTPUT_FILE_NAME}' saved.")
        
        # distribution of new labels
        print(df_labeled['new_label'].value_counts())

    except PermissionError:
        print(f"\nerror '{OUTPUT_FILE_NAME}' cannot save file.")
    except Exception as e:
        print(f"\nerror: {e}")