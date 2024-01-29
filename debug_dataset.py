import matplotlib.pyplot as plt 

def debug_dataset(samples):
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
