import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from .config import (
    EPOCHS_AE,
    LR,
    STYLE_DIM,
    BATCH_SIZE,
    VAL_RATIO,
    RANDOM_STATE,
)

class StyleAutoencoder(nn.Module):
    """
    곡 스타일 임베딩을 추출하기 위한 Autoencoder.
    Encoder: D -> 128 -> STYLE_DIM
    Decoder: STYLE_DIM -> 128 -> D
    """
    def __init__(self, in_dim, bottleneck_dim=STYLE_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def _make_loader(feats_np, shuffle_flag):
    """
    feats_np: (N, D)
    DataLoader를 만들어준다. (입력=타겟 동일)
    """
    class _DS(torch.utils.data.Dataset):
        def __init__(self, arr):
            self.arr = arr.astype(np.float32)
        def __len__(self):
            return self.arr.shape[0]
        def __getitem__(self, idx):
            x = torch.tensor(self.arr[idx], dtype=torch.float32)
            return x, x

    ds = _DS(feats_np)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_flag,
        drop_last=False,
    )
    return dl


def train_autoencoder_for_emotion(
    emotion_id,
    features_np,
    device,
):
    """
    한 감정(emotion_id)에 해당하는 곡 feature들(features_np: (N,D))로
    AE를 학습한다.
    train/val을 나눠서 train_recon_loss / val_recon_loss를 출력한다.
    """
    in_dim = features_np.shape[1]
    model = StyleAutoencoder(in_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # train/val split
    if features_np.shape[0] > 1:
        X_tr, X_val = train_test_split(
            features_np,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    else:
        # 만약 감정 그룹에 곡이 1개 이하라면 split이 무의미하므로 그대로 사용
        X_tr = features_np
        X_val = features_np

    train_loader = _make_loader(X_tr, shuffle_flag=True)
    val_loader   = _make_loader(X_val, shuffle_flag=False)

    for epoch in range(EPOCHS_AE):
        # --- train ---
        model.train()
        train_running_loss = 0.0
        train_total = 0

        for x, x_tgt in train_loader:
            x = x.to(device)
            x_tgt = x_tgt.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x_tgt)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * x.size(0)
            train_total += x.size(0)

        train_loss = train_running_loss / max(train_total, 1)

        # --- val ---
        model.eval()
        val_running_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for x, x_tgt in val_loader:
                x = x.to(device)
                x_tgt = x_tgt.to(device)

                x_hat, _ = model(x)
                vloss = criterion(x_hat, x_tgt)

                val_running_loss += vloss.item() * x.size(0)
                val_total += x.size(0)

        val_loss = val_running_loss / max(val_total, 1)

        print(
            f"[Emotion {emotion_id}] "
            f"Epoch {epoch+1}/{EPOCHS_AE} "
            f"train_recon_loss={train_loss:.6f} "
            f"val_recon_loss={val_loss:.6f}"
        )

    return model
