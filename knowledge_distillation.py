# Python 3.10.12
import numpy # 1.26.4
import sklearn.model_selection # 1.3.2
import keras # 2.15.0
import pickle
import os

TEST_SIZE = 10_000
VAL_SIZE = 10_000
SHAPE = (28, 28, 1)
CLASSES = 10
EPOCHS = 20
VERBOSE = 2

def Teacher(shape=SHAPE, classes=CLASSES, name='teacher'):
    inputs = keras.Input(shape=shape, name='inputs')
    x = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', name='conv_1')(inputs)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(x)
    x = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(32, activation='relu', name='fc_1')(x)
    x = keras.layers.Dropout(0.1, name='dropout')(x)
    outputs = keras.layers.Dense(classes, name='fc_2')(x)
    return keras.Model(inputs, outputs, name=name)

def Student(shape=SHAPE, classes=CLASSES, name='teacher'):
    inputs = keras.Input(shape=shape, name='inputs')
    x = keras.layers.Flatten(name='flatten')(inputs)
    x = keras.layers.Dense(32, activation='relu', name='fc_1')(x)
    x = keras.layers.Dropout(0.1, name='dropout')(x)
    outputs = keras.layers.Dense(classes, name='fc_2')(x)
    return keras.Model(inputs, outputs, name=name)

def Clone(model, name='clone'):
    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    clone._name = name
    return clone

class Distiller(keras.Model):
    def __init__(self, student, teacher, name='distiller'):
        super().__init__(name=name)
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, student_loss, distillation_loss, metrics, temperature, alpha):
        super().compile(optimizer=optimizer, metrics=metrics)
        self._loss_tracker = keras.metrics.Mean(name='loss')
        self.student_loss = student_loss
        self.distillation_loss = distillation_loss
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, x, y, y_pred, sample_weight):
        teacher_pred = self.teacher(x, training=False)
        distillation_loss = self.distillation_loss(
            keras.activations.softmax(teacher_pred / self.temperature),
            keras.activations.softmax(y_pred / self.temperature)
        ) * (self.temperature ** 2)
        student_loss = self.student_loss(y, y_pred)
        loss = (1 - self.alpha) * distillation_loss + self.alpha * student_loss
        self._loss_tracker.update_state(loss)
        return loss

    def call(self, x):
        return self.student(x)

if __name__ == '__main__':
    # Create directories
    os.makedirs('./results/data')
    os.makedirs('./results/teacher')
    os.makedirs('./results/student')

    # Load subsets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Join subsets
    x_train = numpy.concatenate((x_train, x_test))
    y_train = numpy.concatenate((y_train, y_test))

    # Preprocess data
    x_train = numpy.expand_dims(x_train, -1)
    x_train = numpy.divide(x_train, 255, dtype='float32')
    y_train = numpy.squeeze(y_train)

    # Split data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=TEST_SIZE, stratify=y_train)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=VAL_SIZE, stratify=y_train)

    # Save data
    with open('./results/data/x_test.pkl', 'wb') as file:
        pickle.dump(x_test, file)

    with open('./results/data/y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)

    # Create Teacher
    teacher = Teacher()

    # Train Teacher
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    history = teacher.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_data=(x_val, y_val)
    )

    # Save Teacher
    with open('./results/teacher/history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    with open('./results/teacher/weights.pkl', 'wb') as file:
        pickle.dump(teacher.get_weights(), file)

    # Freeze Teacher
    teacher.trainable = False

    # Create Student
    student = Student()

    # Clone Student
    clone = Clone(student)

    # Train Student
    clone.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    history = clone.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_data=(x_val, y_val)
    )

    # Save Student
    with open('./results/student/history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    with open('./results/student/weights.pkl', 'wb') as file:
        pickle.dump(clone.get_weights(), file)

    # Grid search
    results = {}
    for temperature in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        results[temperature] = {}
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            results[temperature][alpha] = {}

            if VERBOSE > 0:
                print(f'temperature={temperature}, alpha={alpha}')

            # Clone Student
            clone = Clone(student)

            # Create Distiller
            distiller = Distiller(clone, teacher)

            # Distill Student
            distiller.compile(
                optimizer=keras.optimizers.Adam(),
                student_loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                distillation_loss=keras.losses.KLDivergence(),
                metrics=[keras.metrics.SparseCategoricalAccuracy()],
                temperature=temperature,
                alpha=alpha
            )
            history = distiller.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                verbose=VERBOSE,
                validation_data=(x_val, y_val)
            )

            # Update results
            results[temperature][alpha]['history'] = history.history
            results[temperature][alpha]['weights'] = distiller.student.get_weights()

    # Save results
    with open('./results/results.pkl', 'wb') as file:
        pickle.dump(results, file)
