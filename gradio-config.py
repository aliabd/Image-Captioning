from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import sequence, image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, \
    RepeatVector, Activation, Flatten, Bidirectional, concatenate
from tensorflow.keras import Input

import numpy as np
import gradio
import pickle

model = InceptionV3(weights='imagenet')
new_input = model.input
hidden_layer = model.layers[-2].output
model_new = Model(new_input, hidden_layer)

unique = pickle.load(open('unique.p', 'rb'))
word2idx = {val:index for index, val in enumerate(unique)}
idx2word = {index:val for index, val in enumerate(unique)}

def preprocess(img):
    # img = image.load_img(image_path, target_size=(299, 299))
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

def load():
    image_model = Sequential([
        Dense(300, input_shape=(2048,), activation='relu'),
        RepeatVector(40)
    ])
    caption_model = Sequential([
        Embedding(8256, 300, input_length=40),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])
    # image_in = Input(shape=(2048,))
    # caption_in = Input(shape=(8256))
    merged = concatenate([image_model.output, caption_model.output], axis=1)
    latent = Bidirectional(LSTM(256, return_sequences=False))(
        merged)
    out_1 = Dense(8256, activation='softmax')(latent)
    out = Activation('softmax')(out_1)
    final_model = Model([image_model.input, caption_model.input], out)

    final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                        metrics=['accuracy'])

    # print(final_model.summary())

    final_model.load_weights("./weights/time_inceptionV3_1.5987_loss.h5")
    return final_model


def predict(image, final_model):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=40,
                                          padding='post')
        e = encode(image)
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > 40:
            break

    return ' '.join(start_word[1:-1])


INPUTS = gradio.inputs.ImageIn(cast_to="pillow")
OUTPUTS = gradio.outputs.Textbox()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load, capture_session=True)

INTERFACE.launch(inbrowser=True)
