"""Convert PyTorch model to ONNX format - improved version."""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple


class ImprovedNet(nn.Module):
    """Improved neural network for MNIST-like classification."""
    
    def __init__(self, input_channels: int = 1, input_height: int = 28, input_width: int = 28, num_classes: int = 10) -> None:
        """Initialize the network architecture.
        
        Args:
            input_channels: Number of input channels
            input_height: Input image height
            input_width: Input image width  
            num_classes: Number of output classes
        """
        super(ImprovedNet, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers."""
        # After conv1: (32, 28, 28)
        # After pool1: (32, 14, 14)
        # After conv2: (64, 14, 14) 
        # After pool2: (64, 7, 7)
        return 64 * 7 * 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def convert_to_onnx() -> bool:
    """Convert PyTorch model to ONNX format.
    
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        print("🔄 Creating improved ONNX model...")
        
        model = ImprovedNet()
        model.eval()
        
        dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
        
        output_dir = Path("..") / "model_repository" / "simple-net" / "1"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.onnx"
        
        print(f" Output path: {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=13,
            input_names=["input"],      # Clear input name
            output_names=["output"],    # Clear output name
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True
        )
        
        print(f" Model successfully converted!")
        
        # Verify file was created
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f" File created, size: {file_size} bytes")
            
            # Validate model
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(" ONNX model validation passed")
            
            # Print model info
            print(" Model Info:")
            print(f"  - Input: {onnx_model.graph.input[0].name}")
            print(f"  - Output: {onnx_model.graph.output[0].name}")
            print(f"  - Input shape: [batch, 1, 28, 28]")
            print(f"  - Output shape: [batch, 10]")
            
            return True
        else:
            print(" Model file was not created")
            return False
            
    except Exception as e:
        print(f" Error converting model: {e}")
        return False


if __name__ == "__main__":
    success = convert_to_onnx()
    if success:
        print(" New ONNX model created successfully!")
    else:
        print(" Failed to create model")