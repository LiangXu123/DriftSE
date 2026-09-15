import torch

def rearrange(tensor, pattern, **kwargs):
    s = tensor.shape
    if pattern == 'b c f t -> b c f 1 t':
        return tensor.unsqueeze(3)
    elif pattern == 'b c f pad t -> b c (f pad) t':
        return tensor.view(s[0], s[1], s[2]*s[3], s[4])
    elif pattern == 'b c fpad t -> (b c t) fpad':
        # B, C, F, T -> B, C, T, F -> B*C*T, F
        return tensor.permute(0, 1, 3, 2).contiguous().view(-1, s[2])
    elif pattern == 'bct fpad -> bct 1 fpad':
        return tensor.unsqueeze(1)
    elif pattern == '(b c t) 1 fpad -> b c fpad t':
        B, C, T = kwargs['b'], kwargs['c'], kwargs['t']
        # B*C*T, 1, F -> B, C, T, F -> B, C, F, T
        return tensor.view(B, C, T, s[2]).permute(0, 1, 3, 2).contiguous()
    elif pattern == 'b (reim c) f t -> b c f t reim':
        return tensor.view(s[0], 2, s[1]//2, s[2], s[3]).permute(0, 2, 3, 4, 1).contiguous()
    elif pattern == 'b g cg f t -> b (g cg) f t':
        return tensor.view(s[0], s[1]*s[2], s[3], s[4])
    elif pattern == 'b (g cg) f t -> b g cg f t':
        g = kwargs['g']
        return tensor.view(s[0], g, s[1]//g, s[2], s[3])
    elif pattern == 'b g t -> b g 1 1 t':
        return tensor.view(s[0], s[1], 1, 1, s[2])
    else:
        raise ValueError(f"Unknown pattern {pattern}")
