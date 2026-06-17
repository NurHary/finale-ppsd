from preprocessing.streamer import VideoStreamer
import vidpath

import random
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt


GC_MAX_VIDEO_PER_FOLDER = 150
GC_TRAIN_RATIO = 0.8

torch.backends.cudnn.benchmark = True

LIST_DIR = [
    vidpath.GC_CLEARVIDPATH,
    vidpath.GC_NEURALVIDPATH,
    vidpath.GC_FFVIDPATH,
    vidpath.GC_FACESHIFTVIDPATH,
    vidpath.GC_FACESWAPVIDPATH,
    vidpath.GC_DFVIDPATH,
    vidpath.GC_DFDVIDPATH,
]


# FUCKING INHERITANCE
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.eye = CNNBranch()
        self.nose = CNNBranch()
        self.temporal = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 7))

        self.fc = nn.Sequential(
            nn.Linear(4101, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, eye, nose, feat):
        B, T, C, H, W = eye.shape
        eye = eye.reshape(B * T, C, H, W)
        nose = nose.reshape(B * T, C, H, W)
        feat = feat.reshape(B * T, -1)
        eye = self.eye(eye)
        nose = self.nose(nose)
        x = torch.cat([eye, nose, feat], dim=1)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        x = x.reshape(B, T, -1)
        x = self.temporal(x)

        return x.mean(1)


def _collectVideos(folder_path, label):
    out = []
    for f in os.listdir(folder_path):
        if f.endswith(".mp4"):
            out.append((os.path.join(folder_path, f), label))
    return out


def makeTensor(sample):
    eye = np.array([(x["left_eye"] + x["right_eye"]) / 2 for x in sample])
    nose = np.array([x["nose"] for x in sample])
    feat = np.array(
        [
            [
                x["feature"]["ear_left"],
                x["feature"]["ear_right"],
                x["feature"]["yaw"],
                x["feature"]["eye_texture"],
                x["feature"]["nose_texture"],
            ]
            for x in sample
        ],
        dtype=np.float32,
    )
    eye = torch.tensor(eye, dtype=torch.float32)
    nose = torch.tensor(nose, dtype=torch.float32)
    feat = torch.tensor(feat, dtype=torch.float32)
    eye = eye.permute(0, 3, 1, 2) / 255
    nose = nose.permute(0, 3, 1, 2) / 255
    feat = (feat - feat.mean(0, keepdim=True)) / (feat.std(0, keepdim=True) + 1e-6)
    return (eye, nose, feat)


def buildSplit():
    train = []
    test = []
    random.seed(42)
    for label, folder in enumerate(LIST_DIR):
        vids = _collectVideos(folder, label)
        random.shuffle(vids)
        vids = vids[:GC_MAX_VIDEO_PER_FOLDER]
        split = int(len(vids) * GC_TRAIN_RATIO)
        train += vids[:split]
        test += vids[split:]

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def trainBatch(model, eye, nose, feat, y, lossfn, opt, scaler):
    eye = eye.cuda(non_blocking=True)
    nose = nose.cuda(non_blocking=True)
    feat = feat.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)

    with torch.autocast("cuda"):
        out = model(eye, nose, feat)

        loss = lossfn(out, y)

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


def splitVideos(videos):
    random.shuffle(videos)
    split = int(len(videos) * GC_TRAIN_RATIO)
    train = videos[:split]
    test = videos[split:]
    return train, test


def evaluate(model, testset, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for eye, nose, feat, y in testset:
            out = model(eye.to(device), nose.to(device), feat.to(device))
            pred = out.argmax(1)
            correct += (pred.cpu() == y).sum()
            total += len(y)
    return correct / total


def evaluate(model, videos, device, player, seq_len=8):
    model.eval()
    correct = 0
    total = 0
    n_class = len(LIST_DIR)
    conf = np.zeros((n_class, n_class), dtype=np.int32)
    with torch.no_grad():
        pbar = tqdm(videos, desc="VALIDATE")
        for path, label in pbar:
            try:
                player.open(path)
                player.startStream()
                player.history.clear()
                vote = []
                while not player.isfinished:
                    player.stream()
                    sample = player.getBatch(size=seq_len)
                    if sample is None:
                        continue
                    eye, nose, feat = makeTensor(sample)
                    eye = eye.unsqueeze(0).to(device)
                    nose = nose.unsqueeze(0).to(device)
                    feat = feat.unsqueeze(0).to(device)
                    out = model(eye, nose, feat)
                    pred = out.argmax(1).item()
                    vote.append(pred)
                if len(vote):
                    pred = max(
                        set(vote),
                        key=vote.count,
                    )
                    total += 1
                    if pred == label:
                        correct += 1
                    conf[label, pred] += 1
                    pbar.set_postfix(acc=f"{100 * correct / max(total, 1):.2f}%")
            except Exception:
                pass
            finally:
                player.history.clear()
                if player.cap:
                    player.cap.release()
    acc = correct / max(total, 1)
    return (acc, conf)


def metrics(conf):
    cls = conf.shape[0]
    print("\n===== METRICS =====")
    for i in range(cls):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        print(
            f"""
CLASS {i}
Precision: {precision:.3f}
Recall   : {recall:.3f}
F1       : {f1:.3f}
"""
        )
    print("\nConfusion Matrix:\n")
    print(conf)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)
    model = TemporalMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

    scaler = torch.amp.grad_scaler.GradScaler()
    weights = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float32).cuda()
    lossfn = nn.CrossEntropyLoss(weight=weights)
    player = VideoStreamer()
    EPOCH = 20
    BATCH_SIZE = 16
    SEQ_LEN = 8
    STRIDE = 12
    train_history = []
    val_history = []
    train_videos, test_videos = buildSplit()
    print("TRAIN:", len(train_videos))
    print("TEST :", len(test_videos))
    for ep in range(EPOCH):
        model.train()
        total_loss = 0
        total_batch = 0
        pbar = tqdm(
            train_videos,
            desc=f"Epoch {ep + 1}",
        )
        for path, label in pbar:
            try:
                player.open(path)
                player.startStream()
                player.history.clear()
                batch_eye = []
                batch_nose = []
                batch_feat = []
                batch_label = []
                step = 0
                while not player.isfinished:
                    player.stream()
                    step += 1
                    if step % STRIDE:
                        continue
                    sample = player.getBatch(size=SEQ_LEN)
                    if sample is None:
                        continue
                    eye, nose, feat = makeTensor(sample)
                    batch_eye.append(eye)
                    batch_nose.append(nose)
                    batch_feat.append(feat)
                    batch_label.append(label)
                    if len(batch_eye) < BATCH_SIZE:
                        continue
                    loss = trainBatch(
                        model,
                        torch.stack(batch_eye),
                        torch.stack(batch_nose),
                        torch.stack(batch_feat),
                        torch.tensor(
                            batch_label,
                            dtype=torch.long,
                        ),
                        lossfn,
                        opt,
                        scaler,
                    )
                    total_loss += loss
                    total_batch += 1
                    pbar.set_postfix(
                        loss=f"{loss:.4f}",
                        avg=f"{total_loss / max(total_batch, 1):.4f}",
                    )
                    batch_eye.clear()
                    batch_nose.clear()
                    batch_feat.clear()
                    batch_label.clear()
            except Exception as e:
                print(
                    "\nSKIP:",
                    os.path.basename(path),
                    e,
                )
            finally:
                player.history.clear()
                if player.cap:
                    player.cap.release()
        avg = total_loss / max(
            total_batch,
            1,
        )
        print(f"\nEPOCH {ep + 1} DONE | LOSS {avg:.4f}")

        print("\nRUN TEST")
        acc, conf = evaluate(model, test_videos, device, player)
        train_history.append(avg)
        val_history.append(acc)
        print(
            f"""
        VALIDATION
        Accuracy:
        {acc * 100:.2f}%
        """
        )
        metrics(conf)
        torch.save(
            {
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            },
            "deepfake.pt",
        )
        sched.step(avg)
    plt.figure(figsize=(8, 5))
    plt.plot(train_history)
    plt.plot(val_history)
    plt.legend(["Loss", "Accuracy"])
    plt.xlabel("Epoch")
    plt.savefig("training.png")

    print("\nGRAPH SAVED")
    print("\nTRAIN FINISHED")


if __name__ == "__main__":
    main()
