#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[3]:


model = keras.Sequential([keras.layers.Dense(units=1,input_shape = [1])])
##밀도가 하나 = 하나의 뉴런 


# In[4]:


model.compile(optimizer = 'sgd', loss = "mean_squared_error")
# 최적화하는 loss 함수 


# In[5]:


xs = np.array([-1.0,0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0,-1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)


# In[13]:


model.fit(xs, ys, epochs = 500)
# epochs는 training 루프를 500번 통과한다는 것이다. (통과하며 손실함수 최소화)


# In[15]:


print(model.predict([10.0]))
# 10을 넣고 예측값을 구해보는 형식 

## 이렇게 할시 19가 아닌 19에 근접한 값이 나온다 >> 데이터가 적어 꼭 2x-1이라는 보장이 없다.


# # 1-2 주차

# In[ ]:


# 데이터를 다운로드 한다 (fashion recognition MNIST)
# fashion mnist 데이터는 keras에서 끌어올 수 있다.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_labels) = fashion_mnist.load_data()

# API로 KERAS 데이터베이스에서 가져오는것 
# 이후 각각 훈련 증명 데이터 , 라벨로 바꿈 


# In[ ]:


# 신경망 정의를 작성해본다.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation =tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
# 여기에는 3개의 layer가 있다. 마지막은 10개의 뉴런을 만들었다. (10개의 옷 class가 존재해서 10개로 만듬)
# 첫번째는 Flatten 레이어로 우리가 넣을 이미지가 28x28이라 이렇게 함
# (28x28) 이미지를 하나의 간단한 선형array로 만들어준다.

# 중간은 hidden layer로 1


# ## work book

# In[16]:


mnist = keras.datasets.fashion_mnist


# In[22]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[23]:


import matplotlib.pyplot as plt
plt.imshow(train_images[0])
print(train_images[0])
print(train_labels[0])


# In[24]:


import matplotlib.pyplot as plt
plt.imshow(train_images[42])
print(train_images[42])
print(train_labels[42])


# In[ ]:


## 255 사이로 pixel이 array 형태로 보여지고 있다.


# In[25]:


train_images = train_images / 255.0
test_images = test_images / 255.0

# 신경망 데이터는 0~255가 아닌 정규화 데이터로 더 잘 돌아가니 정규화를 시켜준다.
# 0~1사이로 된다.


# In[26]:


# 모델을 디자인 해본다.
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation =tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

##  앞서 언급했듯이 flatten은 input layer를 마지막 layer는 각 클래스를 구별해주고
## 중간 hidden layer는 Rules를 찾아내준다. 


# In[32]:


model.compile(optimizer = tf.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs= 10)

# model을 compile을 해서 손실함수를 찾아내고 최적화한다. 목표는 input과 output 사이에
# 어떤 관계가 있나 확인하고 loss가 얼마나 잘 측정됐나 확인하고 최적화한다.


# In[31]:


# test 데이터와 label을 이용해 모델을 평가해본다.
model.evaluate(test_images,test_labels)
# 


# In[33]:


# 각 test_images들의 클래스분류 집합을 만들어본다. 
classifications = model.predict(test_images)

print(classifications[0])
# 밑에 리스트에 나타난 숫자들의 의미는 무엇일까?
# >> 10개 클래스 각각의 확률을 나타낸다. 

## 리스트 안 첫번째 숫자는 티셔트 label 0일 확률이고 매우 적다는 것을 알 수 있다.


# In[40]:


print(test_labels[7])


# - 0	T-shirt/top
# - 1	Trouser
# - 2	Pullover
# - 3	Dress
# - 4	Coat
# - 5	Sandal
# - 6	Shirt
# - 7	Sneaker
# - 8	Bag
# - 9	Ankle boot

# ## 추가 Exercise

#  ### 중간 Hidden layer의 뉴런수를 512, 1024로 늘릴시 training 과정은 오래걸리지만 정확도는 점점 좋아진다. 
# - 뉴런수 증가 >> 더 정확한 계산과 과정시간 증가 (but 모든 경우 좋아지는 것은 아니다.)

# ### Flatten 함수를 input layer에서 사용하지 않을경우 에러가 뜬다. > Data shape에 대해 오류가 발생한다. 신경망의 첫번째 input layer가 데이터와 동일한 모양이어야한다. 우리 데이터는 28x28이미지고 28개의 뉴런과 28개의 레이어는 실현 불가능 하기 때문에 28,28을 784x1로 평활(FLATTEN)하는 것이 타당하다.

# ### Hidden layer를 하나 더 추가할시 어떻게 될까? >> 이 데이터들은 상대적으로 간단한 데이터들이라 큰 변화가 없다. (but 점점 더 복잡해지는 데이터에선 extra layer는 자주 필요하다.)
# - tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(256, activation=tf.nn.relu) 이런식

# ### Epochs를 더더욱 늘려가면 loss 가 감소하는것이 멈추고 오히려 증가하는 'overfitting' 이 있을 수 있다.  그리고 또한 그것을 기다리는 동안 큰 시간이 소비될 수 있다. 어떻게 해야할까?
# - 내가 원하는 loss 값에 도달했을시 멈출 수 있는 방법이 있다. 그럼 원하는 epoch때 멈출 수 있다. 이는 callback 함수를 만들어 시도할 수 있다. 

# In[ ]:


import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])


# In[ ]:


import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


# # 1-3 주차

# ## 합성공 신경망(Convolution Neural Network)
# - 이미지 처리에 탁월한 성능을 보이는 신경망. 기존 신경망은 이미지가 대상이 휘어지거나 이동하거나 방향이 틀어지는 다양한 변형이 있을시 다층 Perceptron은 픽셀이 달라져 민감하게 예측에 영향을 받는다는 단점이 있었다. 이에 공간적인 구조정보 개념을 이용 어떤 픽셀들끼리 연관이 있고 값이 비슷한지 등을 보존하면서 학습할 수 있는 방법이 합성공 신경망이다.
# - 합성공 연산의 결과가 활성화 함수 ReLu를 지난다. 커널과 필터라는 nxm 크기의 행렬로 높이 x 너비크기의 이미지를 처음부터 끝까지 겹치며 훑어서 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곱해서 모두 더한값을 출력하는 것을 의미한다. 주로 3x3, 5x5를 사용하며 예를 들어 각 filter(커널)의 3x3 행렬을 곱해서 주변 3x3 이미지 픽셀을 곱해서 값을 Feature map(특성맵) 3x3의 첫번째값을 구한다. >> 이렇게 방향이 틀어지거나 휘어져도 주변값들이 비슷해서 같은 값이 나오게 된다.
# - Color 이미지에 이용된다. 이유는 흑백이미지는 채널수가 1인데 반해 칼라 이미지는 적생, 녹생, 청색 채널수가 3개이다. (삼원색의 조합) 그래서 3차원 텐서로 표현됩니다. 
# ## 풀링 (Pooling)
# - 특성 맵을 다운샘플링 해서 특성맵의 크기를 줄인다. 일반적으로 최대풀링과 평균 풀링이 사용된다. 예시에서는 Biggest value를 사용하는 최대풀링을 사용해서 줄였다. 

# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[ ]:


# Convolution과 Pooling의 예시 

model = tf.keras.models.Sequential({
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',
                          input_shape = (28,28,1)),
    # 64 keras filter 고 3x3 로 만들고 relu를 이용해 음의값을 나오지 않게 한다. 
    # input data shape 를 입력해주는 데 우리는 흑백 grey 데이터라 채널을 1로 설정해준다.
   # output shape는 (None,26,26,64)가 된다. 이유는 원래 28x 28일시 양쪽끝 pixel 들이 제외된다.
    ## 이유는 양쪽끝 부분 몇개는 pixel들은 주위 3x3을 만족하는 neighbor들이 없어서.
    tf.keras.layers.MaxPooling2D(2,2),
    # max pooling은 최대값을 이용해서 Pooling을 진행하는 것이고 2x2 matrix로 만든다.
    # OUTPUT shape는 (None,13,13,64)가 된다. ㄷ
    # 4x4를 2x2로 줄이는 과정이다.
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    ## 위에서 만들어진 Conv > pooling 데이터를 다시한번 Conv > pooling을 진행해 매우 작게 만든다.
    # Conv 과정에서 output은 (None,11,11,64) pooling에서는 (None,5,5,64)가 된다.
    
    tf.keras.layers.Flatten(),
    ## Flatten에서 Dense layer로 진행할때 이미 우린 quartering + quartering을 진행해서
    ## 이미 매우 작아진 상태이다. 그래서 simplify, 구체화된다.
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
})

# model.summary()를 통해 앞의 과정과 output등을 알 수 있다.


# In[ ]:


## Fashion classifier를 좀더 improving 해보자!!


# In[42]:


import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)
## 이렇게 돌려도 사실 loss가 0.29~~대로 나쁘지 않은 결과를 가져온다.
## test는 다소 loss가 높아지지만 괜찮은급이다.


# In[47]:


# CONVOLUTION과 POOLING을 적용 후 

import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
## RESHAPE를 한 이유는 first convolution은 single tensor를 원하기 때문에 28x28x1의 60000개의 데이터 대신
## 하나의 4D LIST 60000X28X28X1을 사용한 것이다. 이 과정을 안할시 Error가 된다. convolution이 모양을 인식할 수없다.
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)
## training 시간이 확연히 느려졌으나 이는 모든 이미지에 64개의 convolution이 시도되었고 pooling되고 다시 
## 64개의 convolution이 되고 그것이 DNN을 통해 전달되는 현상이라 그렇다. 그래도 loss 값등이 상당히 괜찮은 것을 알수 있다. 개선되었다.


# ## Visualizing Convolutions과 Pooling
# - 아래 코드들은 convolution을 그래프화 해준다. 
# - 우선 test_label의 100개를 추출해보니 0, 23, 28 index가 9로 슈즈인것을 알 수 있다. (ankle boots) 이것들을 이용해 공통된 특징을 알아본다. 

# In[48]:


print(test_labels[:100])


# In[58]:


import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 5
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)


# ### 또다른 예제
# - convolution을 패키지가 아닌 직접 만들어보는 과정을 본다.

# In[63]:


import cv2
import numpy as np
from scipy import misc
i = misc.ascent()
# misc 라이브러리를 scipy모듈에서 가져온다. 우리가 다룰수 있는 좋은 사진을 얻어올 수 있다.


# In[64]:


import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()


# In[67]:


i_transformed = np.copy(i)
# 이미지를 copy한다. 
## 이미지가 숫자 배열로 저장되므로 해당 배열을 복사하기만 하면 변환된 
## 이미지를 만들 수 있습니다. 나중에 반복할 수 있도록 이미지의 size도 파악해 봅시다.
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
print(size_x)
print(size_y)


# In[71]:


i[1,1]


# In[106]:


# filter를 3x3으로 만들어본다. 
# This filter detects edges nicely. 모서리를 잘 찾아낸다. 
# 날카로운 모서리와 직선만 통과하는 convolution을 만든다..

#Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
# 이 필터를 적용하면 특정 feature만 필터를 통과했다는 것을 알 수 있다.

# A couple more filters to try for fun!
#filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# 이 필터는 수직선을 잘 포착한다. 그래서 이것을 실행하고 결과를 보면 우리는 이미지의 수직선이 
# 통과했다는 것을 알 수 있다. 이는 이사진은 단지 상하로 직선이 아니라 이미지 자체의
# 관점 안에서 수직적인 관점을 가지고 있음을 나타낸다.
filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# 이 필터는 수평선을 잘 포착한다. 그림을 보면 많은 수평선들이 그것을 통과했다는것을 알 수 있다. 

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1


# In[107]:


for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        convolution = 0.0
        
        convolution = convolution + (i[x - 1, y-1] * filter[0][0])
        convolution = convolution + (i[x, y-1] * filter[0][1])
        convolution = convolution + (i[x + 1, y-1] * filter[0][2])
        convolution = convolution + (i[x-1, y] * filter[1][0])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x+1, y] * filter[1][2])
        convolution = convolution + (i[x-1, y+1] * filter[2][0])
        convolution = convolution + (i[x, y+1] * filter[2][1])
        convolution = convolution + (i[x+1, y+1] * filter[2][2])
        convolution = convolution * weight
        # 0보다 작을경우와 255보다 클경우만 0과 255로 보정해준다.  
        if(convolution<0):
            convolution=0
        if(convolution>255):
            convolution=255
        i_transformed[x, y] = convolution


# In[108]:


i_transformed


# In[109]:


# Plot the image. Note the size of the axes -- they are 512 by 512
# convolution의 효과를 확인해본다.
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()   


# In[110]:


new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()      
    


# # 1-4 주차
# - Image generator에 관련된 내용 (이미지 사이즈가 크고 각자 다른 크기를 가진 객체들은 어떻게 구별할 수 있을까?)

# In[111]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255)
# rescale을 통해 데이터를 정규화를 시킨다. 
train_generator = train_datagen.flow_from_directory(
train_dir,
    # 디렉토리를 설정 
target_size = (300,300),
    # 신경망을 훈련시키기 위해서는 input data는 같은 사이즈여야만 하기 때문데 
    # 이미지를 resize해야한다.
batch_size = 128,
    # 
class_mode = 'binary')
# 우리는 binary classifier를 이용한다. 말과 사람


# flow_from directory 메소드를 호풀하여 해당 디렉토리와 하위 디렉토리에서 이미지를 로드하도록
# 할 수 있습니다. 사람들이 generator를 sub directory에 poiting 하는것은 흔한 실수입니다.

test_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size = (300,300),
    # 신경망을 훈련시키기 위해서는 input data는 같은 사이즈여야만 하기 때문데 
    # 이미지를 resize해야한다. image가 load 되면서 크기가 조정된다.
    # 이로인해 소스 데이터에 영향을 주지않고 다양한 크기로 테스트를 할 수 있다.
batch_size = 128,
    # image를 하나씩 수행하는것보다 효과적으로 batch로 raining, valud 를 load시킨다.
class_mode = 'binary')


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    # 칼라이미지라 shape 300,300,3으로 둔다. 채널이 3개 
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
    # 1뉴런 존재하고 2개의 class가 존재한다. 1개의 클래스가 0 나머지클래스일때 1을 나타낸다. 
    
])

from tensorflow.keras.optimizers import RMSprop

model.complie(loss= 'binary_crossentropy',
              # 요번엔 두가지 클래스가 존재하니 앞에 쓴 categori~~ crossentropy가 아닌
              # binary cross entropy loss function 사용 
             optimizer = RMSprop(lr = 0.001),
              # Adam optimizer가 아닌 RMSprop을 사용 lr을 learning rate를 의미 
             metrics = ['acc'])

#trainning 단계이다. 앞과 다르게 model.fit이 아닌 model.fit_generator 이용 (dataset에서 generator를 사용했기 때문에)
history = model.fit_generator(
train_generator,
    # training generator로 training 디렉토리에서 이미지를 stream한다.
    # 만들때 사용한 batch size를 기억하는것이 중요하다. 
steps_per_epoch = 8,
    #training 디텍토리에는 1024개의 이미지가 있어한번에 128개의 이미지를 batch size로 
    # load 하고 있습니다. 그래서 모두 load하기위해 8개의 batch를 필요로한다.
    # steps_per_epoch가 이것을 커버한다. 
epochs = 15,
    
validation_data = validation_generator,
    # validation generator를 설정한다.
validation_steps =8,
verbose = 2)
# verbose는 training 진행중에 표시할 양을 구체화한다. 2로 설정하면 우리는
# epoch progress(epoch 진행)을 숨기는 애니메이션을 조금 덜 얻을 수 있다. 


# ## 실습 
# - colab에선 돌아가나 jupyter에서는 오류발생 코드 참고만 할것

# In[128]:


import wget


# In[133]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip     -O /tmp/horse-or-human.zip')
    

# 사람과 말의 데이터 zip파일을 다운로드한다. 


# In[137]:


import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
# 다운로드 작업 이후 이 가상시스템의 temp 디렉토리에 압축을 푼다. 
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()


# In[ ]:


# Directory with our training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

# zip 파일은 두 폴더가 존재한다. 사람과 말을 각각 filtered 한것이다.


# In[ ]:


train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# 몇가지 변수를 지정하고 우리는 파일 이름을 출력하며 파일을 탐색할 수 있습니다.
# 이것들은 라벨을 생성하는데 사용될 수 있습니다. 하지만 keras generator를 이용하면 그럴
# 필요가 없다. 이 데이터를 사용하지 않기를 원하면 파일 이름은 라벨이 있어야한다.


# In[ ]:


print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# 데이터 셋에서 몇개의 임의의 이미지를 표시할 수 있습니다. 
# 여덟명과 여덞마리의 말사진이 나온다. 


# In[ ]:


# 모델 구축


# In[ ]:


import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 이미지 소스가 상당히 크고 300x300 이기 때문에 꽤많은 convolution이 있다. 
# 결국 마지막까지 할시 7x7까지 줄여진다. 


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


# In[ ]:


# data preprocessing


# - 원본 폴더에서 사진을 읽고, 32개(float32)의 텐서로 변환하여(label과 함께) 네트워크에 공급하는 데이터 생성기를 설정해 보겠습니다. training용 이미지와 valid용 이미지용 generator가 하나씩 있습니다. 우리의 생성기는 300x300 크기의 이미지와 그 lavel(이진)의 배치를 산출할 것이다.
# 
# - 이미 알고 계시겠지만, 신경망에 들어가는 데이터는 일반적으로 네트워크에 의한 처리의 용이성을 높이기 위해 어떤 방식으로든 정규화되어야 합니다. (원시 픽셀을 컨브넷에 공급하는 경우는 흔치 않습니다.) 이 경우 픽셀 값을 [0, 1] 범위로 정규화하여 이미지를 사전 처리합니다(원래 모든 값은 [0, 255] 범위임).

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
# 하위 디렉토리가 두개였기 때문에 두개의 클래스에 할당한 것이다.
# (폴더가 두개로 나뉘어져 있었다.)


# In[ ]:


#training
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)


# In[ ]:


# Running the Model 

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
 
# 새로운 데이터에 잘 구분을 할 수 있나 확인을 하기 위하여 pixa에서 데이터를 끌어온다.
# file sytem에 다운받아서 데이터로 사용한다. 
# 이코드를 실행하면 파일을 선택할 수 있고 사람, 말 사진을 넣으면 각 구별된 class가 나온다.
# 한번에 여러파일도 가능하다. 


# In[ ]:


# 중간 과정표현을 시각화하기
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    
# 이 코드로 인간의 이미지가 convolution 된것과 다리와 같은 특징을 볼 수 있다. 머리카락등은 매우 특이하다.


# In[ ]:


# cleanup 
# 다음 exercise를 실행하기 전에 다음셀을 실행하여 커널을 종료하고 메모리 리소스를 확보하십시오.


# In[ ]:


import os, signal
os.kill(os.getpid(), signal.SIGKILL)

