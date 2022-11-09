from itertools import product
from typing import List, Tuple

import torch
import torch.utils.benchmark as bench
from torch import Tensor, jit, nn


class MyLSTM(nn.Module):
    def __init__(self, inp_k, out_k) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(inp_k, out_k)

    def forward(self, xs, state):
        hx, cx = state[0].squeeze(), state[1].squeeze()
        output = []
        for x in xs:
            hx, cx = self.cell(x, (hx, cx))
            output.append(hx)
        return torch.stack(output, dim=0), (hx, cx)


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state[0].squeeze(), state[1].squeeze()
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def forward(model, seq, state):
    with torch.no_grad():
        model(seq, state)


def backward(model, seq, state):
    out, _ = model(seq, state)
    loss = nn.functional.mse_loss(out[-1], seq[0])
    loss.backward()
    return loss.detach()


def main():
    fn = "backward"
    device = torch.device("cuda")

    results = []
    for (T, B, K) in product((64, 128), (4, 64, 128), (128, 256, 512)):
        data = torch.randn((T, B, K), device=device)

        models = {
            "nn.LSTM": nn.LSTM(K, K).to(device),
            "LSTM-Cell": MyLSTM(K, K).to(device),
            "LSTM-Cell JIT": LSTMLayer(LSTMCell, K, K).to(device),
        }

        for name, model in models.items():
            state = (
                torch.randn((1, B, K), device=device),
                torch.randn((1, B, K), device=device),
            )
            results.append(
                bench.Timer(
                    stmt=f"{fn}(model, data, state)",
                    setup="from __main__ import forward, backward",
                    globals={"model": model, "data": data, "state": state},
                    label=f"{fn.upper()} time",
                    sub_label=f"B={B:3d},T={T:3d},K={K:3d}",
                    description=name,
                ).blocked_autorange(min_run_time=2)
            )
    compare = bench.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()


if __name__ == "__main__":
    main()
