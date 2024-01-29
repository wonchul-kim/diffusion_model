import torch 
import numpy as np
from torch.utils.data import Dataset

class SpriteDataset(Dataset):
    beta_1 = 1e-4
    beta_T = 0.02
    T = 500

    # beta는 첨자 1부터 T까지 사용하기 위해 제일 앞에 더미 데이터 tf.constant([0.])를 추가하여 만듬
    beta = torch.cat([ torch.tensor([0]), torch.linspace(beta_1, beta_T, T)], axis=0)
    alpha = 1 - beta
    alpha_bar = torch.exp(torch.cumsum(torch.log(alpha), axis=0))

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x_0 = self.data[i]

        # normalize -1~1로 만들기
        if self.transform:
            x_0 = self.transform(x_0)

        # noise 추가
        t = np.random.randint(1, self.T + 1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(self.alpha_bar[t]) * x_0 + torch.sqrt(1 - self.alpha_bar[t]) * eps

        return x_0, x_t, eps, t

if __name__ == '__main__':
    train_dataset = SpriteDataset(sprites, transform)

    m = 128

    train_loader = DataLoader(train_dataset, batch_size=m, shuffle=True)
    train_loader_iter = iter(train_loader)
    samples = next(train_loader_iter)

    if debug:

        x_0s = samples[0][:6].numpy()
        x_ts = samples[1][:6].numpy()
        epss = samples[2][:6].numpy()
        ts =  samples[3][:6].numpy()

        fig, axs = plt.subplots(figsize=(10,5), nrows=3, ncols=6)

        i = 0
        for (x_0, x_t, eps, t) in zip(x_0s, x_ts, epss, ts):
            x_0 = x_0.transpose(1,2,0)
            x_0 = ((x_0 - x_0.min()) / (x_0.max() - x_0.min())).clip(0,1)
            axs[0][i].imshow(x_0)
            axs[0][i].set_title(f"t={t}")
            axs[0][i].set_xticks([])
            axs[0][i].set_yticks([])

            eps = eps.transpose(1,2,0)
            eps = ((eps - eps.min()) / (eps.max() - eps.min())).clip(0,1)
            axs[1][i].imshow(eps)
            axs[1][i].set_xticks([])
            axs[1][i].set_yticks([])

            x_t = x_t.transpose(1,2,0)
            x_t = ((x_t - x_t.min()) / (x_t.max() - x_t.min())).clip(0,1)
            axs[2][i].imshow(x_t)
            axs[2][i].set_xticks([])
            axs[2][i].set_yticks([])

            if i == 0:
                axs[0][i].set_ylabel('x_0')
                axs[1][i].set_ylabel('eps')
                axs[2][i].set_ylabel('x_t')

            i += 1

        plt.savefig("./assets/dataset.png")
