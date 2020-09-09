# 导入ReduceLROnPlateau
from keras.callbacks import ReduceLROnPlateau

# 定义ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# # 使用ReduceLROnPlateau
# model.fit(X_train, Y_train, callbacks=[reduce_lr])