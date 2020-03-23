from .load_data import load_data
from .emotions import EMOTIONS, OBSERVED_EMOTIONS

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    
    print(f'Features extracted: {x_train.shape[1]}')

    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500,
    )

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    print(y_pred)