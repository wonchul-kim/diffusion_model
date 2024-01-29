import numpy as np
from torchvision import transforms
import torch
from sprite_dataset import SpriteDataset
from ddpm import DDPM

import plotly.graph_objects as go

debug = False
device = 'cuda'
sprites = np.load('sprites_1788_16x16.npy')
print(sprites.shape, sprites.min(), sprites.max())

transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

train_dataset = SpriteDataset(sprites, transform)

m = 8

model = DDPM()
model.to(device)

checkpoint = torch.load('./last.pth')
model.load_state_dict(checkpoint)

alpha = train_dataset.alpha.to(device)
alpha_bar = train_dataset.alpha_bar.to(device)
beta = train_dataset.beta.to(device)
T = train_dataset.T

# 샘플링 단계동안 생성된 이미지를 일정 간격마다 저장할 리스트를 준비
interval = 50 # 20 시간 단계마다 한장씩 생성 결과 기록
X = [] # 생성 이미지 저장
saved_frame = [] # 이미지를 저장한 시간 단계를 저장
N = 1 # 모델에 입력할 샘플 개수

H = 16
W = 16
C = 3

# 최초 노이즈 샘플링
x = torch.randn(size=(N, C, H, W)).to(device)

for t in range(T, 0, -1):
    if t > 1:
        z = torch.randn(size=(N,C,H,W)).to(device)
    else:
        z = torch.zeros((N,C,H,W)).to(device)

    t_torch = torch.tensor([[t]]*N, dtype=torch.float32).to(device)
    eps_theta = model(x, t_torch)
    x = (1 / torch.sqrt(alpha[t])) * \
        (x - ((1-alpha[t])/torch.sqrt(1-alpha_bar[t]))*eps_theta) + torch.sqrt(beta[t])*z
    if (T - t) % interval == 0  or t == 1:
        # 현재 시간 단계로 부터 생성되는 t-1번째 이미지를 저장
        saved_frame.append(t)
        x_np = x.detach().cpu().numpy()

        # (N,C,H,W)->(H,N,W,C)
        x_np = x_np.transpose(2,0,3,1).reshape(H,-1,C)
        x_np = ((x_np - x_np.min()) / (x_np.max() - x_np.min())).clip(0,1)
        X.append( x_np*255.0 ) # 0 ~ 1 -> 0 ~ 255

    del t_torch
    del eps_theta
    
X = np.array(X, dtype=np.uint8)

fig = go.Figure(
    data = [ go.Image(z=X[0]) ],
    layout = go.Layout(
        # title="Generated image",
        autosize = False,
        width = 800, height = 400,
        margin = dict(l=0, r=0, b=0, t=30),
        xaxis = {"title": f"Generated Image: x_{T-1}"},
        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    # play button
                    dict(
                        label="Play", method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50, "easing": "quadratic-in-out"}
                            }
                        ]
                    ),
                    # pause button
                    dict(
                        label="Pause", method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    )
                ],
                direction="left", pad={"r": 10, "t": 87}, showactive=False,
                x=0.1, xanchor="right", y=0, yanchor="top"
            )
        ], # updatemenus = [
    ), # layout = go.Layout(
    frames = [
        {
            'data':[go.Image(z=X[t])],
            'name': t,
            'layout': {
                'xaxis': {'title': f"Generated Image: x_{saved_frame[t]-1}"}
            }
        } for t in range(len(X))
    ]
)

################################################################################
# 슬라이더 처리
sliders_dict = {
    "active": 0, "yanchor": "top", "xanchor": "left",
    "currentvalue": {
        "font": {"size": 15}, "prefix": "input time:",
        "visible": True, "xanchor": "right"
    },
    "transition": {"duration": 100, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9, "x": 0.1, "y": 0,
    "steps": []
}

for t in range(len(X)):
    slider_step = {
        "label": f"{saved_frame[t]}", "method": "animate",
        "args": [
            [t], # frame 이름과 일치해야 연결됨
            {
                "frame": {"duration": 100, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 100}
            }
        ],
    }

    sliders_dict["steps"].append(slider_step)

fig["layout"]["sliders"] = [sliders_dict]
################################################################################

fig.write_html("gapminder_animation.html")
# fig.show()
