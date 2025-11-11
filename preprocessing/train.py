import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import matplotlib.pyplot as plt

'''
    Add labels(target feature) to test dataset by using Neural network
'''

# train data path
TRAIN_FILE_NAME = './data/labeled_dataset.csv'

# test(for predict) data path
TEST_FILE_NAME = './data/test_set.csv' 

# output file path
OUTPUT_FILE_NAME = './data/test_set_with_label_predictions.csv'

# select features (drop track_name, artists, id, etc..)
feature_cols = ['duration_ms', 'danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']

# label column
LABEL_COLUMN = 'new_label' 

# a number of labels
N_CLASSES = 6 # (Relaxed, Energetic, Tense, Romantic, Calm, Sad)

# data load
if not (os.path.exists(TRAIN_FILE_NAME) and os.path.exists(TEST_FILE_NAME)):
    print(f"error: '{TRAIN_FILE_NAME}' or '{TEST_FILE_NAME}' cannot find file")
    print("check data path")
else:
    df_train = pd.read_csv(TRAIN_FILE_NAME)
    df_test = pd.read_csv(TEST_FILE_NAME)

    print(f"train data: {len(df_train)} rows")
    print(f"test data: {len(df_test)} rows")


    # preprocessing
    print("\n--- preprocessing ---")

    # drop null data
    X_train_df = df_train.dropna(subset=feature_cols)[feature_cols]
    y_train_df = df_train.dropna(subset=feature_cols)[LABEL_COLUMN]

    df_test_clean = df_test.dropna(subset=feature_cols).copy()
    X_test_df = df_test_clean[feature_cols]

    print(f"train samples: {len(X_train_df)}, test samples: {len(X_test_df)}")

    # labelEncoder
    # 'Relaxed' -> 0, 'Energetic' -> 1 ...
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_df)

    # feature scaling
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_final_scaled = scaler.transform(X_test_df) # need prediction

    X_train, X_test, y_train, y_test = train_test_split(
    X_train_scaled, y_train_encoded, test_size=0.25, random_state=42, shuffle=True)

    # keras model setting
    print("\n--- NN model ---")
    
    model = Sequential()
    
    # input Layer (input features: len(feature_cols))
    model.add(Dense(64, input_shape=(len(feature_cols),), activation='relu'))
    model.add(Dropout(0.3)) # Dropout to prevent overfit
    
    # hidden Layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # output Layer
    model.add(Dense(N_CLASSES, activation='softmax'))

    # model compile
    model.compile(
        optimizer='adam',                     
        loss='sparse_categorical_crossentropy', # multi label classfication
        metrics=['accuracy']                 
    )
    
    model.summary()
    

    # train model
    history = model.fit(
        X_train,
        y_train,
        epochs=20,         
        batch_size=32,    
        validation_split=0.2 # vaildation set
    )
    
    print("finish train")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"\n(Loss): {test_loss:.4f}")
    print(f"(Accuracy): {test_accuracy * 100:.2f} %")
    
    # visualization - accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # visualization - loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

    # predict
    predictions_proba = model.predict(X_final_scaled)
    
    # select max(prediction)
    predicted_labels_encoded = np.argmax(predictions_proba, axis=1)
    
    
    # num -> text label
    predicted_labels_text = encoder.inverse_transform(predicted_labels_encoded)
    
    # add predicted label 
    df_test_clean['predicted_label'] = predicted_labels_text
    
    # save to csv
    try:
        df_test_clean.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
        print("\n" + "=" * 50)
        print(f"Success: '{OUTPUT_FILE_NAME}' saved.")
        print("=" * 50)
        
        # label distribution
        print("\ndistribution of predicted label")
        print(df_test_clean['predicted_label'].value_counts())

    except PermissionError:
        print(f"\nerror: '{OUTPUT_FILE_NAME}' cannot save file")
    except Exception as e:
        print(f"\nerror: {e}")