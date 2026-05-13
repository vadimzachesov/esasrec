import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        # ИСПРАВЛЕНО: Заменили Conv1d на Linear, чтобы убить лишние transpose(-1, -2).
        # Видеокарта скажет спасибо.
        self.conv1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        # ИСПРАВЛЕНО: Аналогично
        self.conv2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # ИСПРАВЛЕНО: Убраны transpose, так как используем Linear
        outputs = self.conv1(inputs)
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        # ИСПРАВЛЕНО: Убрано сложение (outputs += inputs) отсюда.
        # Оно перенесено в основной цикл, чтобы не складывать с нормализованным входом.
        return outputs


class SASRec(nn.Module):
    def __init__(self, item_num, maxlen=200, hidden_units=256, num_blocks=2,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            # ИСПРАВЛЕНО: Добавлен batch_first=True, чтобы не делать transpose(0, 1) в цикле
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.apply(self._init_weights)

        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_blocks)])

    def _init_weights(self, module):
        # ИСПРАВЛЕНО: Убран nn.Conv1d из проверки, так как теперь везде Linear
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def stack_weights_adjacent(old_state_dict, new_model_num_blocks):
        new_state_dict = {}
        old_num_blocks = new_model_num_blocks // 2

        for key, weight in old_state_dict.items():
            if 'emb' in key or 'last_layernorm' in key:
                new_state_dict[key] = weight

            elif 'layers' in key or 'layernorms' in key or 'alphas' in key:
                parts = key.split('.')
                block_idx = int(parts[1])

                new_key_1 = key.replace(f'.{block_idx}.', f'.{2 * block_idx}.')
                new_key_2 = key.replace(f'.{block_idx}.', f'.{2 * block_idx + 1}.')

                new_state_dict[new_key_1] = weight.clone()
                new_state_dict[new_key_2] = weight.clone()

        return new_state_dict

    def forward(self, input_ids):
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        tl = seqs.shape[1]
        attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        for i in range(self.num_blocks):
            # ИСПРАВЛЕНО: Убрали transpose(0,1), так как теперь batch_first=True
            # ИСПРАВЛЕНО: LayerNorm применяется к копии (Q), чтобы оригинальный seqs прошел чистым
            Q = self.attention_layernorms[i](seqs)

            # ИСПРАВЛЕНО: В MultiheadAttention передаем нормализованный Q
            mha_out, _ = self.attention_layers[i](Q, Q, Q, attn_mask=attn_mask)

            # ИСПРАВЛЕНО: Residual Connection складывается с оригинальным (не нормализованным) seqs
            seqs = seqs + self.alphas[i] * mha_out

            # ИСПРАВЛЕНО: Аналогичный фикс для FFN. Нормализуем копию...
            seqs_norm = self.forward_layernorms[i](seqs)
            ffn_out = self.forward_layers[i](seqs_norm)

            # ИСПРАВЛЕНО: ...а результат складываем с оригинальным seqs
            seqs = seqs + ffn_out

            seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        return self.last_layernorm(seqs)