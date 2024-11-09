import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb


from .models import load_model, save_model
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 100,
    lr: float = 1e-1,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("road_data/val", shuffle=False, batch_size=batch_size, num_workers=4)

    # Define a loss function (MSE for waypoint prediction)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()
        model.train()

        for batch in train_data:
            # Prepare inputs and target based on model type
            if model_name == "mlp_planner" or model_name == "transformer_planner":
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = model(track_left, track_right)

            elif model_name == "cnn_planner":
                image = batch['image'].to(device)
                waypoints = model(image)

            # Target waypoints
            target_waypoints = batch['waypoints'].to(device)

            # Compute loss (MSE between predicted and target waypoints)
            loss = loss_func(waypoints, target_waypoints)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log training loss
            metrics["train_loss"].append(loss.item())
            logger.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

        # Validation
        with torch.inference_mode():
            model.eval()
            total_val_loss = 0

            for batch in val_data:
                if model_name == "mlp_planner" or model_name == "transformer_planner":
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    waypoints = model(track_left, track_right)

                elif model_name == "cnn_planner":
                    image = batch['image'].to(device)
                    waypoints = model(image)

                target_waypoints = batch['waypoints'].to(device)
                val_loss = loss_func(waypoints, target_waypoints)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_data)
            metrics["val_loss"].append(avg_val_loss)

        # Log average train and validation loss to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()

        # Print progress every 10 epochs
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f}"
            )

    # Save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
