import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
"""[None,784]형태의 부정소숫점으로 이루어진 2차원 텐서로 표현."""
x = tf.placeholder(tf.float32, [None, 784])
"""계산과정에서 텐서가 사용되거나 변경될 수 있기 때문에 모델파라미터 variable을 사용한다."""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""x가 여러입력으로 구성된 2D텐서일 경우를 다루기 위해 x와 W를 곱한다."""
y = tf.nn.softmax(tf.matmul(x, W) + b)
"""교차 엔트로피를 구현하기 위해 새 placeholder를 추가한다."""
y_ = tf.placeholder(tf.float32, [None, 10])
"""교차 엔트로피 구현"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
"""비용 최소화에 어떤 변수가 얼마나 영향을 주는지를 효율적으로 계산.경사 하향법을 이용하여 
교차 엔트로피를 최소화 하도록 명령."""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
"""만든 변수들 초기화"""
init = tf.initialize_all_variables()
"""세션에서 모델을 시작하고, 변수들을 초기화"""
sess = tf.Session()
sess.run(init)

# Learning
"""학습 1000 번 실행"""
for i in range(1000):
	"""학습세트로부터 100개의 무작위 batch들을 가져옴.placeholders를 대체하기 위해
	일괄 데이터에 train_step 피딩 실행(확률적 교육-확룰적 경사하강법)"""
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
"""tf.argmax로 특정한 축을 따라 가장 큰 원소의 색인을 알려줌.tf.equal을 이용해
예측이 실제와 맞았는지 확인."""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
"""부정소숫점으로 캐스팅한 후 평균값을 구한다."""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
"""정확도 확인"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))