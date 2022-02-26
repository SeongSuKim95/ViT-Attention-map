import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    # attentions.size() = [12,1,3,197,197]
    with torch.no_grad():
        for attention in attentions:
            # attention.size() =
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
                # attention_heads_fused.size() = [1,197,197] : 12개의 heads를 따라 max
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False) # indices : 1 dim
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)) # 197
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1) # a.sum(dim=-1).size() = [1,197] --> a의 행별 summation으로 normalize

            result = torch.matmul(a, result) # (197,197) * (197,197), result 는 맨처음에 eye matrix, matrix multiplication 
    
    # Look at the total attention between the class token,
    # and the image patches
    # result.size() = (1,197,197)
    mask = result[0, 0 , 1 :] # mask size = 196, except cls token
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5) # 196 ** (1/2) = 14
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    # (14,14)

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        # attention_layer_name 'attn_drop' == MSA layer
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        print(attention_layer_name)
        for name, module in self.model.named_modules():  # model의 sub module에 대해, 만약 'attn_drop(MSA layer)'이 sub module에 있다면
            if attention_layer_name in name: # 각 layer의 MSA layer에 대해
                module.register_forward_hook(self.get_attention) # 매 layer마다 attention map을 추출하기 위하여 모듈 이름을 이용하여 attention map을 담음
                # register_forward_hook : 해당 layer 다음에 self.get_attention 함수 삽입 (수행 X, 함수만 추가)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor) # register_forward_hook으로 get_attention함수가 추가되어 있는 상태로 input_tensor의 forward가 진행
            # self.attentions list에 attention map들이 append 됨
        # print(len(self.attentions)) == 12
        # 이 map들을 가지고 roll out함수 수행
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)

# attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
# discard_ratio=args.discard_ratio)
# mask = attention_rollout(input_tensor)

