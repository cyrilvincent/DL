from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Convolution2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# LeNet-5
def Le_Cun():
    model = Sequential()
    model.add(Convolution2D(6, (3, 3), activation='relu',input_shape=(32,32,1))) #28*28*6
    model.add(AveragePooling2D((2,2), strides=(2,2))) #14*14*6

    model.add(Convolution2D(16, (3, 3), activation='relu')) #10*10*16
    model.add(AveragePooling2D((2,2), strides=(2,2))) #5*5*16

    model.add(Flatten()) #400
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def train():
    model = Le_Cun()
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    trainset = ImageDataGenerator(rescale=1./255)

    batchSize = 8

    trainGenerator = trainset.flow_from_directory(
            'data/dogsvscats/train',
            target_size=(32, 32),
            class_mode="binary",
            color_mode="grayscale",
            batch_size=batchSize)

    model.fit(
            trainGenerator,
            epochs=30,
    )

    model.save('data/dogsvscats/lecunmodel.h5')
    model.save_weights('data/dogsvscats/lecunmodel-weights.h5')

    # Very bad

if __name__ == '__main__':

    train()

    # model = load_model("data/dogsvscats/cholletmodel.h5")
    # img = load_img("data/dogsvscats/small/validation/dogs/dog.1001.jpg", color_mode="grayscale", target_size=(32, 32))
    # img = img_to_array(img)
    # img *= 1. / 255
    # img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # res = model.predict(img)[0][0] # image 0, output 0
    # s = "Dog"
    # if res < 0.5:
    #     s = "Cat"
    #     res = 1 - res
    # print(f"Prediction: {s} {res*100:.0f}%")





