# Use the model defined in Commai following repo :
# https://github.com/commaai/research/blob/master/train_steering_model.py
def get_commai_model(time_len=1):

	ch, row, col = 3, 160, 320  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	          input_shape=(ch, row, col),
	          output_shape=(ch, row, col)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model