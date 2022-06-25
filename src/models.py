import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================
#               Transformer Model
# =============================================

class CreditsEmbedder(nn.Module):
    def __init__(self, features, embedding_projections):
        super(CreditsEmbedder, self).__init__()

        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        self._pos_emb = nn.Embedding(59, 6)

    def forward(self, features):
        batch_size = features[0].shape[0]
        seq_len = features[0].shape[1]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]

        # pos_emb = torch.arange(1, seq_len + 1) * torch.ones(batch_size, seq_len).int()
        # pos_emb = self._pos_emb(pos_emb.cuda())
        # embeddings.append(pos_emb)

        concated_embeddings = torch.cat(embeddings, dim=-1)
        return concated_embeddings

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class TransformerCreditsModel(nn.Module):

    def __init__(self, features, embedding_projections, nhead=3, nlayers=4, p_dropout=0.1, top_classifier_units=32):
        super(TransformerCreditsModel, self).__init__()
        self._credits_embedder = CreditsEmbedder(features, embedding_projections)

        self._d_model = sum([embedding_projections[x][1] for x in features])

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=p_dropout
        )

        self._transformer_enc = nn.TransformerEncoder(
            self._encoder_layer, 
            num_layers=nlayers
        )

        self._top_classifier = nn.Linear(in_features=2 * self._d_model, 
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)

    def forward(self, features):
        emb = self._credits_embedder(features)
        out = self._transformer_enc(emb.permute(1, 0, 2))
        
        out_max_pool = out.permute(1, 0, 2).max(dim=1)[0]
        out_avg_pool = out.permute(1, 0, 2).sum(dim=1) / out.permute(1, 0, 2).shape[1] 
        combined_input = torch.cat([out_max_pool, out_avg_pool], dim=-1)
        classification_hidden = self._top_classifier(combined_input)

        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)
        
        return logits.squeeze(1)


# =================================================
#               LSTM based Model
# =================================================

def get_rnn_model(rnn_type, **kwargs):
    model_dict = {
        "LSTM" : nn.LSTM,
        "GRU" : nn.GRU
    }
    
    return model_dict[rnn_type](**kwargs)


class CreditsRNN(nn.Module):
    def __init__(self, features, embedding_projections, 
                 rnn_type="GRU", rnn_units=128, rnn_num_layers=1, top_classifier_units=32):
        super(CreditsRNN, self).__init__()
        self._credits_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embedding_projections[feature]) 
             for feature in features]
        )
        
        assert rnn_type in ["GRU", "LSTM"]
        self._rnn = get_rnn_model(
            rnn_type, 
            input_size=sum([embedding_projections[x][1] for x in features]),
            hidden_size=rnn_units,
            num_layers=rnn_num_layers,
            batch_first=True, 
            bidirectional=True
        )
        
        
        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(in_features=4*rnn_units, 
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, 
                               out_features=1)
    
    def forward(self, features):
        batch_size = features[0].shape[0]

        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        hidden_states_seq, _ = self._rnn(concated_embeddings)
        
        # batch_size, seq_leq, 2 * hidden_state
        out_max_pool = hidden_states_seq.max(dim=1)[0]
        out_avg_pool = hidden_states_seq.sum(dim=1) / hidden_states_seq.shape[1] 
        
        combined_input = torch.cat([out_max_pool, out_avg_pool], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)
        
        return logits.squeeze(1)
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(
            num_embeddings=cardinality+add_missing, 
            embedding_dim=embed_size, 
            padding_idx=padding_idx
        )
        
# =============================================================
#               LSTM/GRU model with attention mechanism
# =============================================================


class CreditsAttentionRNN(nn.Module):
    def __init__(self, features, embedding_projections, 
                 rnn_type="GRU", rnn_units=128, rnn_num_layers=1, top_classifier_units=64):
        super(CreditsAttentionRNN, self).__init__()
        self._credits_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embedding_projections[feature]) 
             for feature in features]
        )
        
        assert rnn_type in ["GRU", "LSTM"]
        self._rnn = get_rnn_model(
            rnn_type, 
            input_size=sum([embedding_projections[x][1] for x in features]),
            hidden_size=rnn_units,
            num_layers=rnn_num_layers,
            batch_first=True, 
            bidirectional=True
        )
        
        
        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(in_features=6*rnn_units, 
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, 
                               out_features=2)
        
    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context
    
    def forward(self, features):
        batch_size = features[0].shape[0]

        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        hidden_states_seq, final_hidden_state = self._rnn(concated_embeddings)
        
        # [batch_size, seq_leq, 2 * hidden_state]
        out_max_pool = hidden_states_seq.max(dim=1)[0]
        out_avg_pool = hidden_states_seq.sum(dim=1) / hidden_states_seq.shape[1] 
        
        final_hidden_state = final_hidden_state.permute(1, 0, 2).view(batch_size, -1)
        attn_context = self.attention_net(hidden_states_seq.permute(1, 0, 2), final_hidden_state.permute())
        
        combined_input = torch.cat([out_max_pool, out_avg_pool, attn_context], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)
        
        return logits
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(
            num_embeddings=cardinality+add_missing, 
            embedding_dim=embed_size, 
            padding_idx=padding_idx
        )
