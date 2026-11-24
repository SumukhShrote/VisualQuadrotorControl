from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

from run import (
    OracleMLP,
    load_historical_module,
    load_data,
    predict_batched,
    metrics,
    INDICES,
    ROOT,
)

torch.manual_seed(7501)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hist = load_historical_module()
data, true_states, actions, train, val, test = load_data(hist)

aux_path = ROOT / (
    "checkpoints/corners_tcn_student_v1/"
    "stage4_aux13_ft/stage4_aux13_ft_best.pt"
)
aux_ckpt = torch.load(aux_path, map_location="cpu", weights_only=False)

assert list(aux_ckpt["auxiliary_target_indices"]) == INDICES

policy = hist.build_policy(device)
policy.student.load_state_dict(aux_ckpt["student_state_dict"], strict=True)
policy.eval()

head = hist.AuxiliaryStateHead().to(device)
head.load_state_dict(aux_ckpt["auxiliary_state_head_dict"], strict=True)
head.eval()

aux_mean = aux_ckpt["auxiliary_target_mean"].to(device)
aux_std = aux_ckpt["auxiliary_target_std"].to(device)

print("[DATA] Computing estimated states", flush=True)

estimated = []

with torch.no_grad():
    for batch in data["observations"].float().split(2048):
        latent = policy.encode_temporal_state(batch.to(device))
        estimate = head(latent) * aux_std + aux_mean
        estimated.append(estimate.cpu())

estimated = torch.cat(estimated)

assert estimated.shape == (len(actions), 13)
assert torch.isfinite(estimated).all()

mean = estimated[train].mean(0)
std = estimated[train].std(0).clamp_min(1e-5)

x = ((estimated - mean) / std).clamp(-15, 15)

scale = actions[train].std(0).clamp_min(0.05).to(device)

model = OracleMLP().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-5,
)

loader = DataLoader(
    TensorDataset(x[train], actions[train]),
    batch_size=2048,
    shuffle=True,
    pin_memory=True,
)

output = ROOT / (
    "experiments/state_bottleneck/results/"
    "actor_on_estimated13.pt"
)

if output.exists():
    raise FileExistsError(f"Preserving existing model: {output}")

best_loss = float("inf")
best_state = None
stale = 0

for epoch in range(1, 71):
    model.train()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        prediction = model(xb)

        loss = (((prediction - yb) / scale) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()

    pred = predict_batched(model, x[val], device)

    val_loss = (
        ((pred - actions[val]) / scale.cpu()) ** 2
    ).mean().item()

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        stale = 0

        torch.save({
            "model": {
                k: v.detach().cpu()
                for k, v in best_state.items()
            },
            "feature_mean": mean,
            "feature_std": std,
            "feature_indices": INDICES,
            "epoch": epoch,
            "validation_loss": best_loss,
            "input_type": "aux13_estimated_state",
        }, output)
    else:
        stale += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"[TRAIN] epoch={epoch} "
            f"val={val_loss:.6f} best={best_loss:.6f}",
            flush=True,
        )

    if stale >= 12:
        break

model.load_state_dict(best_state)
model.eval()

new_pred = predict_batched(model, x[test], device)

# Compare with the old oracle MLP when fed the same estimates.
oracle_ckpt = torch.load(
    ROOT / "experiments/state_bottleneck/results/"
    "oracle13_teacher_bc.pt",
    map_location="cpu",
    weights_only=False,
)

oracle = OracleMLP().to(device)
oracle.load_state_dict(oracle_ckpt["model"], strict=True)
oracle.eval()

oracle_x = (
    (estimated - oracle_ckpt["feature_mean"])
    / oracle_ckpt["feature_std"]
).clamp(-15, 15)

old_pred = predict_batched(oracle, oracle_x[test], device)

print("\n========== HELD-OUT ACTION PREDICTION ==========")

for name, prediction in [
    ("Old oracle on estimated states", old_pred),
    ("New estimated-state actor", new_pred),
]:
    rmse, r2 = metrics(prediction, actions[test])

    print(
        f"[{name}] mean_RMSE={rmse.mean().item():.6f} "
        f"mean_R2={r2.mean().item():.6f}",
        flush=True,
    )

print("[SAVED]", output, flush=True)
print("[DONE] Estimated-state actor training completed", flush=True)
