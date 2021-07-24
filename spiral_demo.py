import torch
from datasets import *
import visualizer
import models
import matplotlib.pyplot as plt

M_model_state_dict = torch.load(
    'noise_models/spiral_M-DYR-H/0.0/best_epoch_20.pth')
LRM_model_state_dict = torch.load(
    'noise_models/spiral_LRM-DYR-H/0.0/best_epoch_49.pth')

M_model = models.SpiralModel()
M_model.load_state_dict(M_model_state_dict)
LRM_model = models.SpiralModel()
LRM_model.load_state_dict(LRM_model_state_dict)

trainset, _, _ = get_spiral_datasets("datasets")

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=True, num_workers=1, pin_memory=True)

x_train, y_train = next(iter(train_loader))

xi = np.arange(-15, 15, 0.1)
xj = np.arange(-15, 15, 0.1)
x_sample = np.array([[j, i] for i in xi for j in xj])
y = M_model(torch.tensor(x_sample))

# get P(Y=1|X)
confidence = torch.transpose(torch.nn.functional.softmax(y, dim=1), 0, 1)[
    1].detach().numpy()
confidence = confidence.reshape((len(xi), len(xj)))
print(confidence)
x, y = np.meshgrid(xi, xj)

plt.pcolormesh(x, y, confidence)

# x_d0_l0 = get_dim(x_train, y_train_class, dim=0, label_class=0)
# x_d1_l0 = get_dim(x_train, y_train_class, dim=1, label_class=0)
# x_d0_l1 = get_dim(x_train, y_train_class, dim=0, label_class=1)
# x_d1_l1 = get_dim(x_train, y_train_class, dim=1, label_class=1)

# plt.title('spiral dataset')
# plt.plot(x_d0_l0, x_d1_l0, '.', label='class 0')
# plt.plot(x_d0_l1, x_d1_l1, '.', label='class 1')
# plt.colorbar()
plt.show()
