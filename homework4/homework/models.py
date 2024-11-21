from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints


        self.mlp = nn.Sequential(
            nn.Linear(n_track * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_waypoints * 2)
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
         # Concatenate track_left and track_right along the second dimension
        track_input = torch.cat((track_left, track_right), dim=1)  # shape (b, n_track * 2, 2)
        
        # Flatten to (b, n_track * 2 * 2)
        track_input = track_input.view(track_input.size(0), -1)
        
        # Pass through MLP
        waypoints = self.mlp(track_input)
        
        # Reshape output to (b, n_waypoints, 2)
        waypoints = waypoints.view(-1, self.n_waypoints, 2)
        
        return waypoints


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Embedding for input track points
        self.track_embedding = nn.Linear(2, d_model)  # Project (x, y) to d_model dimensions

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(n_track * 2, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Query embedding for predicting waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Output layer to map the transformer's output to waypoint coordinates
        self.output_layer = nn.Linear(d_model, 2)

    def forward(self,track_left: torch.Tensor,track_right: torch.Tensor,**kwargs,) -> torch.Tensor:
      """
      Args:
        track_left (torch.Tensor): shape (b, n_track, 2)
        track_right (torch.Tensor): shape (b, n_track, 2)

     Returns:
        torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
      """
      batch_size = track_left.size(0)

      # Concatenate left and right track points along the track dimension
      track = torch.cat([track_left, track_right], dim=1)  # shape (b, n_track * 2, 2)
  
      # Embed track points and add positional encoding
      track_embedded = self.track_embedding(track) + self.positional_encoding  # shape (b, n_track * 2, d_model)

      # Pass through transformer encoder
      track_encoded = self.transformer_encoder(track_embedded.permute(1, 0, 2))  # shape (n_track * 2, b, d_model)

      # Generate query embeddings
      queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # shape (b, n_waypoints, d_model)

      # Compute attention scores (dot product of queries and keys)
      attention_scores = torch.matmul(queries, track_encoded.permute(1, 2, 0))  # shape (b, n_waypoints, n_track * 2)

      # Apply softmax to get attention weights
      attention_weights = torch.softmax(attention_scores, dim=-1)  # shape (b, n_waypoints, n_track * 2)

      # Compute weighted sum of values (encoded track points)
      attended_output = torch.matmul(attention_weights, track_encoded.permute(1, 0, 2))  # shape (b, n_waypoints, d_model)

      # Pass through output layer to predict (x, y) coordinates for waypoints
      waypoints = self.output_layer(attended_output)  # shape (b, n_waypoints, 2)

      return waypoints



class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3, input_size: tuple = (96, 128)):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.input_size = input_size  # Dynamic input size

        # Normalization buffers
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, H/16, W/16)
            nn.ReLU(),
        )

        # Dynamically calculate the flattened size
        self.flattened_size = self.calculate_flattened_size(self.input_size)

        # Define fully connected layers using the dynamically calculated size
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_waypoints * 2),
        )

    def calculate_flattened_size(self, input_size):
        """
        Dynamically calculates the flattened size of the feature maps after convolutional layers.
        """
        # Create a dummy input tensor with the given input size
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
            #print(f"Shape after conv_layers: {output.shape}")  # Debugging print
            flattened_size = output.view(1, -1).size(1)
            #print(f"Calculated flattened size: {flattened_size}")  # Debugging print
        return flattened_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and values in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Normalize the input image
        input_mean = self.input_mean.to(image.device)
        input_std = self.input_std.to(image.device)
        x = (image - input_mean[None, :, None, None]) / input_std[None, :, None, None]

        #print(f"Input image shape: {image.shape}")  # Debugging print

        # Pass through convolutional layers
        x = self.conv_layers(x)
        #print(f"Shape after conv_layers: {x.shape}")  # Debugging print

        # Pass through fully connected layers
        x = self.fc_layers(x)
        #print(f"Shape after fc_layers: {x.shape}")  # Debugging print

        # Reshape to (batch_size, n_waypoints, 2) for (x, y) coordinates
        waypoints = x.view(-1, self.n_waypoints, 2)
        #print(f"Output waypoints shape: {waypoints.shape}")  # Debugging print
        return waypoints



MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

