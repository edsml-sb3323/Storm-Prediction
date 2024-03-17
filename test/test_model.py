import unittest
import torch
from Tools.Model import EncoderDecoderConvLSTM

import sys

sys.path.append("..")


class TestEncoderDecoderConvLSTM(unittest.TestCase):
    def test_forward(self):
        """Test the forward pass of the EncoderDecoderConvLSTM model."""
        # Create an instance of the model
        input_channels = 1
        num_convlstm_en = 2
        num_convlstm_de = 2
        model = EncoderDecoderConvLSTM(input_channels, num_convlstm_en,
                                       num_convlstm_de)

        # Create a sample input tensor
        batch_size = 2
        seq_len = 5
        height, width = 64, 64
        input_tensor = torch.rand(batch_size, seq_len, input_channels,
                                  height, width)

        # Forward pass
        output = model(input_tensor)

        # Check the shape of the output
        self.assertEqual(output.shape, torch.Size([batch_size, 3, 1,
                                                   height, width]))

    def test_init_hidden(self):
        """Test the init_hidden method of ConvLSTM encoders and decoders."""
        # Create an instance of the model
        input_channels = 1
        num_convlstm_en = 2
        num_convlstm_de = 2
        model = EncoderDecoderConvLSTM(input_channels, num_convlstm_en,
                                       num_convlstm_de)

        # Test init_hidden method for ConvLSTM encoders
        enc_hidden = model.convlstm_encoders[0].init_hidden(
            batch_size=2, img_size=(64, 64)
        )
        self.assertEqual(enc_hidden[0].shape, torch.Size([2, 4, 64, 64]))
        self.assertEqual(enc_hidden[1].shape, torch.Size([2, 4, 64, 64]))

        # Test init_hidden method for ConvLSTM decoders
        dec_hidden = model.convlstm_decoders[0].init_hidden(
            batch_size=2, img_size=(64, 64)
        )
        # Update the expected shape to match the actual initialization
        self.assertEqual(dec_hidden[0].shape, torch.Size([2, 4, 64, 64]))
        self.assertEqual(dec_hidden[1].shape, torch.Size([2, 4, 64, 64]))


if __name__ == "__main__":
    unittest.main()
