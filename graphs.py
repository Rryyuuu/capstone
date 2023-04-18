import matplotlib.pyplot as plt
from train_cnn import history

# Plot the accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs , acc , 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('C:/Users/ryu/Desktop/main_data/gis/acc_loss/savefig_default_acc.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('C:/Users/ryu/Desktop/main_data/gis/acc_loss/savefig_default_loss.png')

plt.show()