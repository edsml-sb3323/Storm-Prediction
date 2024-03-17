# Modify from a basic example given by ChatGPT
# https://chat.openai.com/share/065ed735-a8cb-4e3f-ade6-e7ec2772707e

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialise a convolutional Long short-term memory (ConvLSTM) cell.

        Parameters
        ----------
        input_dim: int

            The number of channels of an input tensor.

        hidden_dim: int

            The number of channels of a hidden state.

        kernel_size: int

            The size of a convolutional kernel (must be an odd number).

        bias: bool

            The choice to be chosen whether a bias is included in the model.

        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Perform a forward pass through the ConvLSTM cell.

        Parameters
        ----------
        input_tensor: torch.Tensor

            The input tensor for the current time step.

        cur_state: list of torch.Tensor

            A list containing the current hidden state and cell state.

        Returns
        -------
        torch.Tensor

            The output hidden state for the current time step.

        torch.Tensor

            The output cell state for the current time step.

        """
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, img_size):
        """
        Initialise the hidden state for the first time step.

        Parameters
        ----------
        batch_size: int

            The size of the input batch.

        img_size: tuple

            A tuple containing the height and width of the input image.

        Returns
        -------
        torch.Tensor

            The initial hidden state.

        torch.Tensor

            The initial cell state.

        """
        height, width = img_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(
        self,
        input_chan,
        num_convlstm_en,
        num_convlstm_de,
        num_output_channels=3,  # Predicting 3 images
        kernel_size=3,
    ):
        """
        Initialise an encoder-decoder convolutional LSTM.

        Parameters
        ----------
        input_chan: int

            The number of channels of the input.

        num_convlstm_en: int

            The number of ConvLSTM encoders.

        num_convlstm_de: int

            The number of ConvLSTM decoders.

        num_output_channels: int

            The number of channels of the output.
            As we are predicting 3 images, so num_output_channels=3 by default.

        kernel_size: int

            The size of the convolutional kernel
            (width=height by design, and must be an odd number)

        """
        super(EncoderDecoderConvLSTM, self).__init__()

        # ConvLSTM encoders
        self.convlstm_encoders = nn.ModuleList()
        self.convlstm_encoders.append(
            ConvLSTMCell(
                input_dim=input_chan, hidden_dim=4, kernel_size=kernel_size,
                bias=True
            )
        )
        current_dim = 4
        for i in range(1, num_convlstm_en):
            self.convlstm_encoders.append(
                ConvLSTMCell(
                    input_dim=current_dim,
                    hidden_dim=current_dim * 2,
                    kernel_size=kernel_size,
                    bias=True,
                )
            )
            current_dim *= 2

        # ConvLSTM decoders
        self.convlstm_decoders = nn.ModuleList()
        for i in range(num_convlstm_de - 1):
            self.convlstm_decoders.append(
                ConvLSTMCell(
                    input_dim=current_dim,
                    hidden_dim=current_dim // 2,
                    kernel_size=kernel_size,
                    bias=True,
                )
            )
            current_dim = current_dim // 2
        self.convlstm_decoders.append(
            ConvLSTMCell(
                input_dim=current_dim, hidden_dim=3, kernel_size=kernel_size,
                bias=True
            )
        )

        # 3D-CNN decoders
        self.cnn_decoders = nn.Sequential(
            nn.Conv3d(
                in_channels=3, out_channels=3, kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=3,
                out_channels=num_output_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
        )

    def autoencoder(
        self,
        X,
        seq_len,
        future_step,
        h_t_convlstm_encoder_list,
        c_t_convlstm_encoder_list,
        h_t_convlstm_decoder_list,
        c_t_convlstm_decoder_list,
    ):
        """
        Parameters
        ----------
        X: torch.Tensor

            An input tensor with a shape of
            (batch_size, sequence_length, channels, height, width).

        seq_len: int

            The length of the input sequence.

        future_step: int

            The number of future steps to predict.

        h_t, h_t2, h_t3, h_t4, h_t5: torch.Tensors

            Hidden states (h_t) for different ConvLSTM layers.

        c_t, c_t2, c_t3, c_t4, c_t5: torch.Tensors

            Cell states (c_t) for different ConvLSTM layers.

        Returns
        -------
        outs: list of torch.Tensors

            The list of predicted outputs for each future step.

        """
        outs = []

        # ConvLSTM encoders
        for t in range(seq_len):
            (
                h_t_convlstm_encoder_list[0],
                c_t_convlstm_encoder_list[0],
            ) = self.convlstm_encoders[0](
                input_tensor=X[:, t, :, :],
                cur_state=[h_t_convlstm_encoder_list[0],
                           h_t_convlstm_encoder_list[0]],
            )
            for i in range(1, len(self.convlstm_encoders)):
                (
                    h_t_convlstm_encoder_list[i],
                    c_t_convlstm_encoder_list[i],
                ) = self.convlstm_encoders[i](
                    input_tensor=h_t_convlstm_encoder_list[i - 1],
                    cur_state=[
                        h_t_convlstm_encoder_list[i],
                        h_t_convlstm_encoder_list[i],
                    ],
                )

        # ConvLSTM decoders
        # Take the h_t produced by the last ConvLSTM encoder as input
        for t in range(future_step):
            (
                h_t_convlstm_decoder_list[0],
                c_t_convlstm_decoder_list[0],
            ) = self.convlstm_decoders[0](
                input_tensor=h_t_convlstm_encoder_list[-1],
                cur_state=[h_t_convlstm_decoder_list[0],
                           h_t_convlstm_decoder_list[0]],
            )
            for i in range(1, len(self.convlstm_decoders)):
                (
                    h_t_convlstm_decoder_list[i],
                    c_t_convlstm_decoder_list[i],
                ) = self.convlstm_decoders[i](
                    input_tensor=h_t_convlstm_decoder_list[i - 1],
                    cur_state=[
                        h_t_convlstm_decoder_list[i],
                        h_t_convlstm_decoder_list[i],
                    ],
                )
            outs.append(h_t_convlstm_decoder_list[-1])  # Predictions

        outs = torch.stack(outs, 1)
        outs = outs.permute(0, 2, 1, 3, 4)
        outs = self.cnn_decoders(outs)

        return outs

    def forward(self, X, future_step=1):
        """
        Parameters
        ----------
        X: torch.Tensor

            An input tensor with a shape of
            (batch_size, sequence_length, channels, height, width).

        future_step: int

            The number of future steps to predict.

        Returns
        -------
        out: torch.Tensor

            The final predicted output of the model.

        """
        b, seq_len, _, h, w = X.size()

        # Initialise h_t and c_t and store them in lists
        h_t_convlstm_encoder_list, c_t_convlstm_encoder_list = [], []
        for e in self.convlstm_encoders:
            h_t, c_t = e.init_hidden(batch_size=b, img_size=(h, w))
            h_t_convlstm_encoder_list.append(h_t)
            c_t_convlstm_encoder_list.append(c_t)
        h_t_convlstm_decoder_list, c_t_convlstm_decoder_list = [], []
        for d in self.convlstm_decoders:
            h_t, c_t = d.init_hidden(batch_size=b, img_size=(h, w))
            h_t_convlstm_decoder_list.append(h_t)
            c_t_convlstm_decoder_list.append(c_t)

        # Autoencoder forward
        out = self.autoencoder(
            X,
            seq_len,
            future_step,
            h_t_convlstm_encoder_list,
            c_t_convlstm_encoder_list,
            h_t_convlstm_decoder_list,
            c_t_convlstm_decoder_list,
        )
        return out
