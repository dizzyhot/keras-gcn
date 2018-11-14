from kegra.utils import *
from kegra.layers.graph import GraphConvolution
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def build_model(X, y, support=1):
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    X_in = Input(shape=(X.shape[1],))
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H] + G)
    # Compile model
    model = Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
    return model


X, A, y = load_karate_fake_data()
# Reload the trained karate model
model = build_model(X, y)
model.load_weights('karate.h5')
A_ = preprocess_adj(A, True)
graph = [X, A_]
y, idx, train_mask = get_karate_splits(y)

preds = model.predict(graph, batch_size=A.shape[0])

print("The shape of preds is ")
# print(preds.shape)
# print(preds)
val_loss = evaluate_preds(preds, [y], [idx])

# print("The value loss would be " + val_loss)
print("The value loss is " + str(val_loss))
model.summary()
